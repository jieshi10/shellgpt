from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
import json
import datetime
from utils import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = AttrDict(dict(
    seed=0,
    multi_gpu=True,
    device='cuda',
    special_cmd_path=None,
    tokenization_with_rules=True,
    gpt_model_path='pretrained-models/gpt2',
    init_ckpt=None,
    model_name=f"models/pretrained-gpt-{datetime.datetime.now().strftime('%y%m%d%H%M%S')}",
    warmup_steps=4000,
    n_epochs=20,
    save_every_n_epochs=5,
    max_length=256,
    clm=AttrDict(dict(  # causal language modeling
        first_epoch=False,
        n_epochs=4,
        batch_size=32,
    )),
    clm_datasets=[
        AttrDict(dict(
            packing='keep_all',  # ['xxxxx', 'yyyyy'] => ['xxx', 'xxy', 'yyy']  drop last
            prepend_special_token=None,
            append_eos_token=False,
            data_paths=[
                'data/manpage.txt'
            ],
        )),
        AttrDict(dict(
            packing='truncate',  # ['xxxxx', 'y', 'z'] => ['xxx', 'yz']  truncate sent which is too long
            prepend_special_token='<sh>',
            append_eos_token=True,
            data_paths=[
                'data/train_sample.txt',
            ],
        )),
        AttrDict(dict(
            packing=None,
            prepend_special_token=None,
            append_eos_token=False,
            data_paths=[
                'data/train_aligned_sample.txt',
            ],
        )),
    ],
    ecl=AttrDict(dict(  # equivalent command learning
        data_paths=[
            'data/filtered_path_train_sample.txt',
        ],
        n_dataloader_workers=4,
        n_epochs=1,
        batch_size=8,
        tau=.1,
    )),
    lr=1e-2,
))

multi_gpu = config.multi_gpu
if multi_gpu:
    from accelerate import Accelerator

set_seed(config.seed)

tokenization_with_rules = config.tokenization_with_rules

model_name = config.model_name

warmup_steps = config.warmup_steps
n_epochs = config.n_epochs
max_length = config.max_length

scheduler_fn = lambda step: 768**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))

def output(accelerator, *args):
    if multi_gpu:
        accelerator.print(*args)
    else:
        print(*args)

def save_model(accelerator, model, model_path):
    if multi_gpu:
        model = accelerator.unwrap_model(model)
        accelerator.save(model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    output(accelerator, 'saved:', model_path)

def to_device(x, device):
    for k in x:
        x[k] = x[k].to(device)

class CmdContrastDataset(Dataset):

    def __init__(self, tokenizer, dataset):
        super(Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        tokenizer = self.tokenizer
        y = self.dataset[index]['y']
        x = self.dataset[index]['x']
        x_prime = x[:]
        op = np.random.randint(3)
        if op == 0:  # shuffle
            np.random.shuffle(x_prime)
            if x_prime == x:
                op = 1
        if op != 0:
            if op == 1:  # delete
                x_prime.pop(np.random.randint(len(x_prime)))
            else:  # insert duplicate
                assert op == 2
                duplicate = x_prime[np.random.randint(len(x_prime))]
                x_prime.insert(np.random.randint(len(x_prime) + 1), duplicate)
        return (do_encode(tokenizer, ['<sh>'] + y, max_length, perform_tokenization=False, padding_right=False),
            do_encode(tokenizer, ['<sh>'] + sum(x, []), max_length, perform_tokenization=False, padding_right=False), 
            do_encode(tokenizer, ['<sh>'] + sum(x_prime, []), max_length, perform_tokenization=False, padding_right=False))

def mean_pooling(x, y, eps=1e-8):
    mask = x['attention_mask']  # (batch_size, seq_len)
    l = torch.sum(mask, dim=-1, keepdim=True)  # (batch_size, 1)
    last_hidden_state = y.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
    mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
    pooling = torch.sum(last_hidden_state * mask, dim=1)  # (batch_size, hidden_size)
    return pooling / (l + eps)  # (batch_size, hidden_size)

def load_clm_dataset(accelerator, tokenizer, config):
    eos_token = tokenizer.eos_token
    training_set = []
    weights = []
    for dataset in config.clm_datasets:
        output(accelerator, 'loading clm dataset:', dataset)
        assert dataset.packing is None or dataset.packing in ['keep_all', 'truncate']

        training_subset = []
        for data_path in dataset.data_paths:
            with open(data_path, encoding='utf-8') as fin:
                for line in fin:
                    if dataset.packing is None or dataset.packing == 'truncate':
                        line = line.strip()
                    training_subset.append(line)

        # training_subset = training_subset[:100]

        training_subset = [do_tokenize(tokenizer, s, tokenization_with_rules=tokenization_with_rules) for s in tqdm(training_subset, desc='Tokenizing')]

        new_max_len = max_length
        if dataset.prepend_special_token is not None:
            new_max_len -= 1
        
        if dataset.append_eos_token:
            assert dataset.packing is not None and dataset.packing != 'keep_all'
            training_subset = [s[:new_max_len-1] + [eos_token] for s in training_subset]

        if dataset.packing is not None and dataset.packing == 'keep_all':
            packed_training_set = []
            rest = []
            for sentence in tqdm(training_subset, desc='Packing'):
                current = rest + sentence
                while len(current) >= new_max_len:
                    packed_training_set.append(current[:new_max_len])
                    current = current[new_max_len:]
                rest = current
            # drop rest == drop last
            training_subset = packed_training_set
        
        elif dataset.packing is not None and dataset.packing == 'truncate':
            packed_training_set = []
            sentence = []
            for s in tqdm(training_subset, desc='Packing'):
                assert len(s) <= new_max_len and len(sentence) <= new_max_len
                if len(sentence) + len(s) <= new_max_len:
                    sentence += s
                else:  # len(sentence) + len(s) > new_max_len
                    if len(sentence) > 0:
                        packed_training_set.append(sentence)
                    sentence = s
            if len(sentence) > 0:
                packed_training_set.append(sentence)
            training_subset = packed_training_set
        
        if dataset.prepend_special_token is not None:
            training_subset = [[dataset.prepend_special_token] + s for s in training_subset]
        
        training_set += training_subset
        weights.append(len(training_subset))
    
    training_set = [do_encode(tokenizer, s, max_length, perform_tokenization=False, padding_right=False, truncate_right=True) for s in tqdm(training_set, desc='Encoding')]

    # for x in training_set:
    #     print(tokenizer.convert_ids_to_tokens(x['input_ids']))
    # exit()

    output(accelerator, weights)

    weights = sum([[1 / len(weights) / w] * w for w in weights], [])
    assert len(training_set) == len(weights)

    # print(weights)
    # print(sum(weights, 0))
    # exit()

    dataloader = DataLoader(training_set, batch_size=config.clm.batch_size,
                            sampler=WeightedRandomSampler(weights, num_samples=len(training_set))
                            # shuffle=True
                        )

    return dataloader

def load_ecl_dataset(accelerator, tokenizer, data_path):
    eos_token = tokenizer.eos_token
    output(accelerator, 'load ecl dataset:', data_path)
    training_set = []
    with open(data_path, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                training_set.append(json.loads(line))
    
    # training_set = training_set[:100]

    training_set = [{
            'x': [do_tokenize(tokenizer, x, tokenization_with_rules=tokenization_with_rules)[:max_length-1] + [eos_token] for x in r['x']],
            'y': do_tokenize(tokenizer, r['y'], tokenization_with_rules=tokenization_with_rules)[:max_length-1] + [eos_token],
        } for r in tqdm(training_set, desc='Tokenizing')]
    output(accelerator, 'size:', len(training_set))
    return training_set

def main():
    if multi_gpu:
        accelerator = Accelerator()
        device = accelerator.device
    else:
        accelerator = None
        device = config.device
    
    # prepare model

    tokenizer = get_gpt_tokenizer(config)
    model = GPT2LMHeadModel.from_pretrained(config.gpt_model_path, output_hidden_states=True)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if config.init_ckpt is not None:
        model.load_state_dict(torch.load(config.init_ckpt, map_location=device), strict=True)
        output(accelerator, 'loaded:', config.init_ckpt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

    # load dataset

    clm_dataloader = load_clm_dataset(accelerator, tokenizer, config)

    ecl_training_set = [load_ecl_dataset(accelerator, tokenizer, data_path) for data_path in config.ecl.data_paths]
    ecl_weights = [[1 / len(ecl_training_set) / len(subset)] * len(subset) for subset in ecl_training_set]
    assert len(ecl_training_set) == len(ecl_weights)
    # print(ecl_weights)
    ecl_training_set = sum(ecl_training_set, [])
    ecl_weights = sum(ecl_weights, [])
    assert len(ecl_training_set) == len(ecl_weights)
    # exit()

    ecl_training_set = CmdContrastDataset(tokenizer, ecl_training_set)

    # for i in [17, ]:
    #     y, x, x_prime = ecl_training_set[i]
    #     print(tokenizer.convert_ids_to_tokens(y['input_ids']))
    #     print(tokenizer.convert_ids_to_tokens(x['input_ids']))
    #     print(tokenizer.convert_ids_to_tokens(x_prime['input_ids']))
    #     print('---')
    # exit()

    # pay attention to `shuffle`, which should be `True`
    ecl_dataloader = DataLoader(ecl_training_set,
        batch_size=config.ecl.batch_size,
        num_workers=config.ecl.n_dataloader_workers,
        sampler=WeightedRandomSampler(ecl_weights, num_samples=len(ecl_training_set)),
        # shuffle=False,
    )
    
    # for batch_y, batch_x, batch_x_prime in ecl_dataloader:
    #     for y, y_mask, x, x_mask, x_prime, x_prime_mask in zip(
    #             batch_y['input_ids'], batch_y['attention_mask'],
    #             batch_x['input_ids'], batch_x['attention_mask'],
    #             batch_x_prime['input_ids'], batch_x_prime['attention_mask']):
    #         print(tokenizer.convert_ids_to_tokens(y))
    #         print(tokenizer.convert_ids_to_tokens(x))
    #         print(tokenizer.convert_ids_to_tokens(x_prime))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y[y_mask.bool()])))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x[x_mask.bool()])))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x_prime[x_prime_mask.bool()])))
    #         print('---')
    #     break
    # exit()

    if multi_gpu:
        model, optimizer, clm_dataloader, ecl_dataloader, scheduler = accelerator.prepare(
            model, optimizer, clm_dataloader, ecl_dataloader, scheduler
        )
    
    # training

    total_steps = 0
    clm_total_steps = 0
    ecl_total_steps = 0
    epoch = 0
    clm_epoch_flag = config.clm.first_epoch
    epoch_counter = 0
    while epoch < n_epochs:
        epoch += 1
        epoch_counter += 1
        if clm_epoch_flag:
            # print(epoch, epoch_counter, 'clm')
            dataloader = clm_dataloader
            clm_total_steps += len(clm_dataloader)
            if epoch_counter == config.clm.n_epochs:
                clm_epoch_flag = False
                epoch_counter = 0
        else:
            # print(epoch, epoch_counter, 'ecl')
            dataloader = ecl_dataloader
            ecl_total_steps += len(ecl_dataloader)
            if epoch_counter == config.ecl.n_epochs:
                clm_epoch_flag = True
                epoch_counter = 0
        total_steps += len(dataloader)
    output(accelerator, 'total_steps:', total_steps)  # ~100,000
    output(accelerator, 'clm_total_steps:', clm_total_steps)
    output(accelerator, 'ecl_total_steps:', ecl_total_steps)

    # exit()

    log_softmax = nn.LogSoftmax(dim=-1)
    cos_sim = nn.CosineSimilarity(dim=-1)

    epoch = 0
    global_steps = 0
    epoch_loss_list = []

    config.total_steps = total_steps
    config.clm_total_steps = clm_total_steps
    config.ecl_total_steps = ecl_total_steps
    config.epoch_loss_list = epoch_loss_list

    clm_epoch_flag = config.clm.first_epoch
    epoch_counter = 0

    model.train()
    with torch.enable_grad():
        while epoch < n_epochs:
            epoch += 1
            epoch_counter += 1
            epoch_loss = 0
            output(accelerator, f'epoch {epoch}:')
            if clm_epoch_flag:
                objective = 'clm'
                dataloader = clm_dataloader
                if epoch_counter == config.clm.n_epochs:
                    clm_epoch_flag = False
                    epoch_counter = 0
                
                for x in tqdm(dataloader):
                    to_device(x, device)
                    loss = model(**x, labels=x['input_ids']).loss
                    if multi_gpu:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    epoch_loss += loss.item()
                    global_steps += 1
            else:
                objective = 'ecl'
                dataloader = ecl_dataloader
                if epoch_counter == config.ecl.n_epochs:
                    clm_epoch_flag = True
                    epoch_counter = 0
                
                for batch_y, batch_x, batch_x_prime in tqdm(dataloader):
                    to_device(batch_y, device)
                    to_device(batch_x, device)
                    to_device(batch_x_prime, device)
                    output_y = model(**batch_y, labels=batch_y['input_ids'])
                    output_x = model(**batch_x, labels=batch_x['input_ids'])
                    loss = output_y.loss + output_x.loss
                    output_x_prime = model(**batch_x_prime, labels=batch_x_prime['input_ids'])
                    pooling_y = mean_pooling(batch_y, output_y)
                    pooling_x = mean_pooling(batch_x, output_x)
                    pooling_x_prime = mean_pooling(batch_x_prime, output_x_prime)
                    pos = cos_sim(pooling_y, pooling_x) / config.ecl.tau  # (batch_size, )
                    neg = cos_sim(pooling_y, pooling_x_prime) / config.ecl.tau  # (batch_size, )
                    pos = pos.unsqueeze(-1)  # (batch_size, 1)
                    neg = neg.unsqueeze(-1)  # (batch_size, 1)
                    scores = torch.hstack([pos, neg])  # (batch_size, 2)
                    pos_scores = log_softmax(scores)[:, 0]  # (batch_size, )
                    loss -= torch.sum(pos_scores)
                    if multi_gpu:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    epoch_loss += loss.item()
                    global_steps += 1
            
            epoch_loss_list.append({
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'global_steps': global_steps,
                'objective': objective,
            })
            output(accelerator, f'epoch_loss: {epoch_loss}')
            
            if epoch % config.save_every_n_epochs == 0 and (not multi_gpu or accelerator.is_main_process):
                with open(model_name + '-' + str(epoch) + '.json', 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
                save_model(accelerator, model, model_name + '-' + str(epoch) + '.pt')

if __name__ == '__main__':
    main()

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import datetime
from utils import *
from copy import deepcopy
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.stem import PorterStemmer
from Levenshtein import ratio

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

config = dict(
    seed=0,
    multi_gpu=False,
    device='cuda',

    # # recommendation
    # train_data_paths=['data/recommendation_train_sample.txt'],
    # test_data_paths=['data/recommendation_test_sample.txt'],
    # lr=2e-3,
    # bos_token='<sh>',
    # sep_token='</s>',
    # do_substitute=True,
    # accurate_hit=False,
    # metric='acc',
    
    # # correction
    # train_data_paths=['data/correction_train_sample.txt'],
    # test_data_paths=['data/correction_test_sample.txt'],
    # lr=1e-2,
    # bos_token='<sh>',
    # sep_token='<i>',
    # do_substitute=False,
    # accurate_hit=False,
    # metric='acc',

    # nl2bash
    train_data_paths=['data/nl2bash/nl2sh_train.txt'],
    test_data_paths=['data/nl2bash/nl2sh_dev.txt'],
    lr=1e-3,
    bos_token='<nl2sh>',
    sep_token='<nl2sh>',
    do_substitute=False,
    accurate_hit=False,
    metric='char-bleu-4',

    special_cmd_path=None,
    tokenization_with_rules=True,
    gpt_model_path='pretrained-models/gpt2',
    init_ckpt='models/pretrained-gpt.pt',
    # init_ckpt=None,
    model_name=f"models/finetuned-gpt-nl2bash-{timestamp}",
    output_file_path=f"outputs/out-finetuned-gpt-nl2bash-{timestamp}",
    warmup_steps=200,
    n_epochs=50,
    eval_every_n_epochs=1,
    eval_batch_size=24,
    output_every_n_epochs=1,
    save_every_n_epochs=1,
    max_length=256,
    batch_size=32,
    num_beams=5,
    num_ret_seq=5,
)
config = AttrDict(config)

multi_gpu = config.multi_gpu
if multi_gpu:
    from accelerate import Accelerator

set_seed(config.seed)

tokenization_with_rules = config.tokenization_with_rules

model_name = config.model_name

warmup_steps = config.warmup_steps
n_epochs = config.n_epochs
max_length = config.max_length

eval_batch_size = config.eval_batch_size
num_ret_seq = config.num_ret_seq

do_substitute = config.do_substitute
accurate_hit = config.accurate_hit

if not accurate_hit:
    from suffix_automaton import SuffixAutomaton

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

def load_dataset(tokenizer, data_path):
    dataset = []
    group = []
    with open(data_path, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                group.append(line)
            else:
                dataset.append(group)
                group = []
    return dataset

def evaluate(accelerator, device, tokenizer, model, test_set, sa, output_file_path=None):
    if output_file_path is not None:
        fout = open(output_file_path, 'w', encoding='utf-8')
    correct = 0
    sum_char_bleu2 = 0
    sum_char_bleu4 = 0
    sum_word_bleu2 = 0
    sum_word_bleu4 = 0
    sum_edit = 0
    model.eval()
    with torch.no_grad(), tqdm(total=len(test_set), desc='Evaluating') as bar:
        for k in range(0, len(test_set), eval_batch_size):
            batch = test_set[k:k + eval_batch_size]
            x = batch_encode(
                tokenizer,
                [x for x, _, _ in batch],
                max_length*2//3,
                padding_max_len=False,
                padding_right=False,
                truncate_right=False,
                perform_tokenization=False)
            # for tokens in x['input_ids']:
            #     print(tokenizer.convert_ids_to_tokens(tokens))
            for k in x:
                x[k] = x[k].to(device)
            y = model.generate(
                **x,
                max_length=max_length+10,
                num_beams=config.num_beams, num_return_sequences=num_ret_seq,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True)
            y = tokenizer.batch_decode(y, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert len(y) == len(batch) * num_ret_seq
            for i, (xxx, xxx_mask, (_, xx, yy)) in enumerate(zip(x['input_ids'], x['attention_mask'], batch)):
                pred = y[i * num_ret_seq:i * num_ret_seq + num_ret_seq]
                xxx = tokenizer.convert_ids_to_tokens(xxx[xxx_mask.bool()])
                xxx = tokenizer.convert_tokens_to_string(xxx)
                res = {
                    'x': xx,
                    'y': yy,
                    'prediction': pred,
                    'verdict': False
                }
                yy = yy[-1]
                res['gold'] = yy
                max_char_bleu2 = 0
                max_char_bleu4 = 0
                max_word_bleu2 = 0
                max_word_bleu4 = 0
                max_edit = 0
                for p in pred:
                    p = p.replace(tokenizer.pad_token, '')
                    assert p.startswith(xxx)
                    if len(p) > len(xxx):
                        if config.sep_token == tokenizer.eos_token:
                            if p.endswith(tokenizer.eos_token):
                                p = p.split(tokenizer.eos_token)[-2]
                            else:
                                p = p.split(tokenizer.eos_token)[-1]
                            p += tokenizer.eos_token
                        else:
                            p = p.split(config.sep_token)[-1]
                            if not p.endswith(tokenizer.eos_token):
                                p += tokenizer.eos_token
                        if do_substitute:
                            p = do_substitution(p)
                        bleu2, bleu4 = sentence_bleu([yy], p, weights=[(1./2., 1./2.), (1./4., 1./4., 1./4., 1./4.)])
                        if bleu2 > max_char_bleu2:
                            max_char_bleu2 = bleu2
                        if bleu4 > max_char_bleu4:
                            max_char_bleu4 = bleu4
                        bleu2, bleu4 = sentence_bleu(
                            [PorterStemmer().stem(w) for w in word_tokenize(yy.lower())],
                            [PorterStemmer().stem(w) for w in word_tokenize(p.lower())],
                            weights=[(1./2., 1./2.), (1./4., 1./4., 1./4., 1./4.)])
                        if bleu2 > max_word_bleu2:
                            max_word_bleu2 = bleu2
                        if bleu4 > max_word_bleu4:
                            max_word_bleu4 = bleu4
                        edit = ratio(yy, p)
                        if edit > max_edit:
                            max_edit = edit
                        if (accurate_hit and yy == p) or (not accurate_hit and sa.is_substring(xx + [p])):
                            correct += 1
                            res['correct'] = p
                            res['verdict'] = True
                            break
                sum_char_bleu2 += max_char_bleu2
                sum_char_bleu4 += max_char_bleu4
                sum_word_bleu2 += max_word_bleu2
                sum_word_bleu4 += max_word_bleu4
                sum_edit += max_edit
                res['metrics'] = {
                    'max-char-bleu-2': max_char_bleu2,
                    'max-char-bleu-4': max_char_bleu4,
                    'max-word-bleu-2': max_word_bleu2,
                    'max-word-bleu-4': max_word_bleu4,
                    'max-edit': max_edit,
                }
                if output_file_path is not None:
                    fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            bar.update(len(batch))
    if output_file_path is not None:
        fout.close()
    return {
        'acc': correct / len(test_set),
        'char-bleu-2': sum_char_bleu2 / len(test_set),
        'char-bleu-4': sum_char_bleu4 / len(test_set),
        'word-bleu-2': sum_word_bleu2 / len(test_set),
        'word-bleu-4': sum_word_bleu4 / len(test_set),
        'edit': sum_edit / len(test_set),
    }

def main():
    if multi_gpu:
        accelerator = Accelerator()
        device = accelerator.device
    else:
        accelerator = None
        device = config.device

    # prepare model

    tokenizer = get_gpt_tokenizer(config)
    eos_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(config.gpt_model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if config.init_ckpt is not None:
        model.load_state_dict(torch.load(config.init_ckpt, map_location=device), strict=True)
        output(accelerator, 'loaded:', config.init_ckpt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)
    
    # load dataset

    training_set = []
    for data_path in config.train_data_paths:
        training_set += load_dataset(tokenizer, data_path)
    
    # print(training_set[:3])
    # print(training_set[-3:])
    # exit()

    test_set = []
    for data_path in config.test_data_paths:
        test_set += load_dataset(tokenizer, data_path)
    
    # training_set = training_set[:1000]
    # test_set = test_set[:1000]

    output(accelerator, 'train:', len(training_set), 'test:', len(test_set))

    # encoding

    def add_special_tokens(dataset):
        return [[
            (config.bos_token if i == 0 else '') + s + config.sep_token
            for i, s in enumerate(group[:-1])
        ] + [group[-1] + eos_token]
        for group in dataset]

    training_set = add_special_tokens(training_set)
    training_set = [(
            do_encode(
                tokenizer,
                ''.join(group),
                max_length,
                tokenization_with_rules=tokenization_with_rules,
                padding_right=False,
                truncate_right=False),
            [do_substitution(s) for s in group] if do_substitute else group
        ) for group in tqdm(training_set, desc='Encoding')]
    
    # print(list(map(lambda x: x[1], training_set[:3])))
    
    # for encoded, substitute in training_set:
    #     input_ids = encoded['input_ids']
    #     attn_mask = encoded['attention_mask']
    #     if input_ids[0] == tokenizer.pad_token_id:
    #         print(substitute)
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids)))
    #         print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[attn_mask.bool()])))
    #         break
    # exit()
    
    test_set = add_special_tokens(test_set)
    test_set = [(y[:-1], y) for y in test_set]
    test_set = [(''.join(gx), gy) for gx, gy in test_set]
    test_set = [(
            do_tokenize(
                tokenizer, x,
                tokenization_with_rules=tokenization_with_rules)[-max_length*2//3:],
            [do_substitution(yy) for yy in gy] if do_substitute else gy
        ) for x, gy in tqdm(test_set, desc='Tokenizing')]
    test_set = [(tx, gy[:-1], gy) for tx, gy in test_set]

    # print(list(map(lambda x: x[2], test_set[:3])))
    
    # print(tokenizer.convert_ids_to_tokens(training_set[0][0]['input_ids']))
    # print(test_set[0][2])

    # exit()

    if accurate_hit:
        sa = None
    else:
        # output(accelerator, 'SuffixAutomaton: Constructing')
        sa = SuffixAutomaton(sum([gy for _, _, gy in test_set] + [gy for _, gy in training_set], []))
        output(accelerator, 'SuffixAutomaton: Constructed')

    # print(sum([gy for _, _, gy in test_set] + [gy for _, gy in training_set], [])[:10])
    # print(sum([gy for _, gy in training_set] + [gy for _, _, gy in test_set], [])[:10])
    
    test_set.sort(key=lambda r: len(r[0]))

    # print([len(test_set[i][0]) for i in range(10)])
    # print([len(test_set[i][0]) for i in range(-10, 0)])
    
    # print(repr(tokenizer.convert_tokens_to_string(test_set[0][0])), repr(test_set[0][1]))
    # print(repr(tokenizer.convert_tokens_to_string(test_set[12][0])), repr(test_set[12][1]))

    dataloader = DataLoader([encoded for encoded, _ in training_set], batch_size=config.batch_size, shuffle=True)
    
    # exit()

    if multi_gpu:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
    
    # training

    total_steps = n_epochs * len(dataloader)
    output(accelerator, 'total_steps:', total_steps)  # ~100,000

    epoch = 0
    global_steps = 0
    epoch_loss_list = []

    config.total_steps = total_steps
    config.epoch_loss_list = epoch_loss_list

    best_model = None
    best_acc = 0

    while epoch < n_epochs:
        epoch += 1
        epoch_loss = 0
        output(accelerator, f'epoch {epoch}:')
        model.train()
        with torch.enable_grad():
            for x in tqdm(dataloader):
                for k in x:
                    x[k] = x[k].to(device)
                # print(x)
                # print(tokenizer.batch_decode(x['input_ids'], clean_up_tokenization_spaces=False))
                # exit()
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
        
        record = {
            'epoch': epoch,
            'epoch_loss': epoch_loss,
            'global_steps': global_steps,
        }
        epoch_loss_list.append(record)
        output(accelerator, f'epoch_loss: {epoch_loss}')
        
        if epoch % config.eval_every_n_epochs == 0 and (not multi_gpu or accelerator.is_main_process):
            output_file_path = None
            if epoch % config.output_every_n_epochs == 0:
                output_file_path = config.output_file_path + '-' + str(epoch) + '.json'
            metrics = evaluate(accelerator, device, tokenizer, model, test_set, sa, output_file_path)
            output(accelerator, 'metrics:\n', json.dumps(metrics, ensure_ascii=False, indent=4))
            record['metrics'] = metrics
            if metrics[config.metric] > best_acc:
                best_acc = metrics[config.metric]
                best_model = deepcopy(model)
                output(accelerator, 'model copied')
        
        if epoch % config.save_every_n_epochs == 0 and (not multi_gpu or accelerator.is_main_process):
            with open(model_name + '-' + str(epoch) + '.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            save_model(accelerator, model, model_name + '-' + str(epoch) + '.pt')
    
    if best_model is not None:
        output(accelerator, 'best:', best_acc)
        output_file_path = config.output_file_path + '-best.json'
        metrics = evaluate(accelerator, device, tokenizer, best_model, test_set, sa, output_file_path)
        output(accelerator, 'metrics:\n', json.dumps(metrics, ensure_ascii=False, indent=4))
        save_model(accelerator, best_model, model_name + '-best.pt')

if __name__ == '__main__':
    main()

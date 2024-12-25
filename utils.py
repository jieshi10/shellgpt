from transformers import GPT2TokenizerFast
import torch
import torch.nn as nn
import numpy as np
import random
import json
import string
import re


# -------------- tokenizer --------------


def get_gpt_tokenizer(config):
    tokenizer = GPT2TokenizerFast.from_pretrained(config.gpt_model_path, add_prefix_space=False)

    special_cmd_path = config.special_cmd_path
    if special_cmd_path is not None:
        with open(special_cmd_path, encoding='utf-8') as fin:
            special_cmd = json.load(fin)
            # special_cmd += ['Ġ' + cmd for cmd in special_cmd]
        tokenizer.add_tokens(special_cmd)

    tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>',

        # 'additional_special_tokens': ['<datetime>', '<digits>', '<hash>']
        
        'additional_special_tokens': [
            '<i>',  # incorrect marker: cmd<i></s> indicates cmd could be incorrect
            # '<align>',  # stage 1: manpage
            '<explain>',  # stage 2: domain specific alignment: explain sh cmd
            # stage 3: multitask finetuning
            '<nl2sh>',  # nl->sh
            '<sh2nl>',  # sh->nl
            '<sh>',  # sh only, can be multiple sh cmd
            '<nl>',  # nl only
        ]
    })

    return tokenizer


def tokenize_cmd(tokenizer, s, tokenization_with_rules=True):
    """
    s = 'cmd' (without special tokens)
    """
    if not tokenization_with_rules:
        return tokenizer.tokenize(s)
    r = []
    w = ''
    for c in s:
        if c == ' ':
            if len(w) > 0:
                r.append(w)
                w = ''
            w += c
        elif c in string.punctuation + string.digits:
            if w == ' ':
                w += c
                r.append(w)
                w = ''
            else:
                if len(w) > 0:
                    r.append(w)
                    w = ''
                r.append(c)
        else:
            if c in string.ascii_uppercase:
                if len(w) > 0:
                    r.append(w)
                    w = ''
            w += c
    if len(w) > 0:
        r.append(w)
    # merge consecutive upper case letters
    r2 = []
    for w in r:
        if len(r2) > 0 and r2[-1][-1] in string.ascii_uppercase and len(w) == 1 and w in string.ascii_uppercase:
            r2[-1] += w
        else:
            r2.append(w)
    return sum([tokenizer.tokenize(w) for w in r2], [])


def do_tokenize(tokenizer, s, tokenization_with_rules=True):
    """
    s = '<s>cmd1</s>cmd2</s><pad><pad>'
    """
    special_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token] + tokenizer.additional_special_tokens
    # code 1:
    s = [s]
    for t in special_tokens:
        r = []
        for ss in s:
            for sss in ss.split(t):
                r.append(sss)
                r.append(t)
            r.pop(-1)
        s = list(filter(lambda x: x != '', r))
    # code 2:
    # r = []
    # cur = ''
    # i = 0
    # while i < len(s):
    #     starts_with_special_token = False
    #     for t in special_tokens:
    #         if s[i:i+len(t)] == t:
    #             starts_with_special_token = True
    #             if cur != '':
    #                 r.append(cur)
    #                 cur = ''
    #             r.append(t)
    #             i += len(t)
    #             break
    #     if not starts_with_special_token:
    #         cur += s[i]
    #         i += 1
    # if cur != '':
    #     r.append(cur)
    # s = r
    return sum(list(map(lambda x: [x] if x in special_tokens else tokenize_cmd(tokenizer, x, tokenization_with_rules=tokenization_with_rules), s)), [])


def do_encode(tokenizer, s, max_len, perform_tokenization=True, padding_max_len=True, tokenization_with_rules=True, padding_right=True, return_tensors=True, truncate_right=True):
    """
    s = 'cmd1</s>cmd2</s>'
    """
    # bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token = tokenizer.pad_token
    # s = [bos_token] + do_tokenize(tokenizer, s)
    if perform_tokenization:
        s = do_tokenize(tokenizer, s, tokenization_with_rules=tokenization_with_rules)
    if len(s) > max_len:
        if truncate_right:
            s = s[:max_len-1]
            mask = [1] * len(s)
            if len(s) >= 1 and s[-1] == eos_token:
                if padding_right:
                    s += [pad_token]
                    mask += [0]
                else:
                    s = [pad_token] + s
                    mask = [0] + mask
            else:
                s += [eos_token]
                mask += [1]
        else:  # truncate left
            s = s[-max_len:]
            mask = [1] * len(s)
            if len(s) >= 1 and s[0] == eos_token:
                if padding_right:
                    s = s[1:] + [pad_token]
                    mask = mask[1:] + [0]
                else:
                    s[0] = pad_token
                    mask[0] = 0
        assert len(s) == max_len
    else:
        mask = [1] * len(s)
    assert len(mask) == len(s)
    if padding_max_len:
        if padding_right:
            s += [pad_token] * (max_len - len(s))
            mask += [0] * (max_len - len(mask))
        else:  # padding left
            s = [pad_token] * (max_len - len(s)) + s
            mask = [0] * (max_len - len(mask)) + mask
        assert len(s) == max_len and len(mask) == max_len
    s = tokenizer.convert_tokens_to_ids(s)
    return {
        'input_ids': torch.tensor(s).long() if return_tensors else s,
        'attention_mask': torch.tensor(mask).long() if return_tensors else mask,
    }


def batch_encode(tokenizer, batch, max_len, tokenization_with_rules=True, padding_max_len=True, padding_right=True, perform_tokenization=True, truncate_right=True):
    """
    batch = [
        'cmd1</s>',
        'cmd1</s>cmd2</s>',
    ]
    """
    input_ids = []
    attention_mask = []
    l = 0
    for s in batch:
        encoded_s = do_encode(
            tokenizer, s, max_len,
            perform_tokenization=perform_tokenization,
            padding_max_len=False,
            tokenization_with_rules=tokenization_with_rules,
            padding_right=padding_right,
            return_tensors=False,
            truncate_right=truncate_right)
        input_ids.append(encoded_s['input_ids'])
        attention_mask.append(encoded_s['attention_mask'])
        l = max(l, len(input_ids[-1]))
    assert 0 < l <= max_len
    if not padding_max_len:
        max_len = l
    if padding_right:
        input_ids = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in input_ids]
        attention_mask = [x + [0] * (max_len - len(x)) for x in attention_mask]
    else:  # padding left
        input_ids = [[tokenizer.pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        attention_mask = [[0] * (max_len - len(x)) + x for x in attention_mask]
    return {
        'input_ids': torch.tensor(input_ids).long(),
        'attention_mask': torch.tensor(attention_mask).long(),
    }


# ---------- cmd split & merge ----------


def split_cmd(tokenizer, s):
    """
    s = "cmd11 cmd12</s>cmd21 cmd22</s><pad><pad></s>"
    returns:
    split_s = ["cmd11", "cmd12", "</s>", "cmd21", "cmd22", "</s>"]
    """
    pad_token = tokenizer.pad_token
    eos_token = tokenizer.eos_token
    s = s.replace(pad_token, '')
    s = s.split(eos_token)
    while s[-1] == '':
        s.pop(-1)
    s = sum([ss.split() + [eos_token] for ss in s], [])
    return s


def merge_cmd(tokenizer, split_s):
    """
    split_s = ["cmd11", "cmd12", "</s>", "cmd21", "cmd22", "</s>"]
    returns:
    s = "cmd11 cmd12</s>cmd21 cmd22</s>"
    """
    eos_token = tokenizer.eos_token
    s = ''
    last = ''
    for ss in split_s:
        if ss == eos_token:
            s += ss
        elif s == '' or last == eos_token:
            s += ss
        else:
            s += ' ' + ss
        last = ss
    return s


def group_split_cmd(tokenizer, split_s):
    """
    split_s = ["cmd11", "cmd12", "</s>", "cmd21", "cmd22", "</s>"]
    returns:
    grouped_s = [["cmd11", "cmd12", "</s>"], ["cmd21", "cmd22", "</s>"]]
    """
    eos_token = tokenizer.eos_token
    grouped_s = []
    cur = []
    for ss in split_s:
        cur.append(ss)
        if ss == eos_token:
            if len(cur) != 0:
                grouped_s.append(cur)
                cur = []
    if len(cur) != 0:
        grouped_s.append(cur)
    return grouped_s


# --- substitution with placeholders ---


def replace_random_str(match):
    s = match.group(0)
    # if s[0] in ['"', "'"]:
    #     s = s[1:]
    # if s[-1] in ['"', "'"]:
    #     s = s[:-1]
    if re.match("^[0-9]+$", s):
        return s
    if re.match("^([0-9]{6,}[A-Za-z]{4,})|([A-Za-z]{4,}[0-9]{6,})$", s):
        return s
    n_digits = 0
    for l, c in enumerate(s, start=1):
        if c in string.digits:
            n_digits += 1
        if l >= 4 and n_digits / l >= 0.5:
            return "<str>"
    return s


def do_substitution(s):
    """
    s = "cmd1</s>cmd2</s><pad><pad></s>"

    - <ip>: x.x.x.x localhost
    - <ip>:<port>: <ip>:80
    - <digits>: 0.1 0-1 0_1 000000
    - <time>: 00:00:00 00:00 00:0 "<digits> <time>"
    """
    s = re.sub(r"[0-9xX]{1,3}(\.[0-9xX]{1,3}){3}", '<ip>', s)
    s = re.sub(r"localhost", '<ip>', s)
    s = re.sub(r"<ip>:[0-9]{2,5}", '<ip>:<port>', s)
    
    s = re.sub(r"[0-9]+([\.\-_][0-9]+)+([\.\-_]\*)*", '<digits>', s)
    s = re.sub(r"[0-9A-Za-z]{6,}", replace_random_str, s)
    s = re.sub(r"[0-9]{6,}([\.\-_][0-9]+)*", '<digits>', s)
    s = re.sub(r"[0-9]{2}:[0-9]{2}:([0-9]{2}|\*)([\.\-_]\*)*", '<time>', s)
    s = re.sub(r"[0-9]{2}:([0-9]?\*|[0-9]{,2})", '<time>', s)
    s = re.sub(r"<digits>[ \.\-_]<time>", '<time>', s)

    s = re.sub("""['"]<digits>['"]""", '<digits>', s)
    s = re.sub("""['"]<time>['"]""", '<time>', s)
    s = re.sub("""['"]<str>['"]""", '<str>', s)
    return s


# ---------------- utils ----------------


# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    config = dict(
        special_cmd_path='data/special_cmd.json',
        gpt_model_path='pretrained-models/gpt2'
    )
    config = AttrDict(config)
    tokenizer = get_gpt_tokenizer(config)

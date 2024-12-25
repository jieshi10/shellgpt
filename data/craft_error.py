from Levenshtein import ratio, distance
import json
import numpy as np
import string

shift_map = {
    '`': '~',
    '1': '!',
    '2': '@',
    '3': '#',
    '4': '$',
    '5': '%',
    '6': '^',
    '7': '&',
    '8': '*',
    '9': '(',
    '0': ')',
    '-': '_',
    '=': '+',
    
    'q': 'Q',
    'w': 'W',
    'e': 'E',
    'r': 'R',
    't': 'T',
    'y': 'Y',
    'u': 'U',
    'i': 'I',
    'o': 'O',
    'p': 'P',
    '[': '{',
    ']': '}',

    'a': 'A',
    's': 'S',
    'd': 'D',
    'f': 'F',
    'g': 'G',
    'h': 'H',
    'j': 'J',
    'k': 'K',
    'l': 'L',
    ';': ':',
    "'": '"',
    '\\': '|',

    'z': 'Z',
    'x': 'X',
    'c': 'C',
    'v': 'V',
    'b': 'B',
    'n': 'N',
    'm': 'M',
    ',': '<',
    '.': '>',
    '/': '?',
}
keyboard_pertub_map = {
    '`': ['1'],
    '1': ['`', '2', 'q'],
    '2': ['1', '3', 'q', 'w'],
    '3': ['2', '4', 'w', 'e'],
    '4': ['3', '5', 'e', 'r'],
    '5': ['4', '6', 'r', 't'],
    '6': ['5', '7', 't', 'y'],
    '7': ['6', '8', 'y', 'u'],
    '8': ['7', '9', 'u', 'i'],
    '9': ['8', '0', 'i', 'o'],
    '0': ['9', '-', 'o', 'p'],
    '-': ['0', '=', 'p', '['],
    '=': ['-', '[', ']'],

    'q': ['w', 'a', '1', '2'],
    'w': ['q', 'e', '2', '3', 'a', 's'],
    'e': ['w', 'r', '3', '4', 's', 'd'],
    'r': ['e', 't', '4', '5', 'd', 'f'],
    't': ['r', 'y', '5', '6', 'f', 'g'],
    'y': ['t', 'u', '6', '7', 'g', 'h'],
    'u': ['y', 'i', '7', '8', 'h', 'j'],
    'i': ['u', 'o', '8', '9', 'j', 'k'],
    'o': ['i', 'p', '9', '0', 'k', 'l'],
    'p': ['o', '[', '0', '-', 'l', ';'],
    '[': ['p', ']', '0', '=', ';', "'"],
    ']': ['[', '=', "'", '\\'],

    'a': ['s', 'q', 'w', 'z'],
    's': ['a', 'd', 'w', 'e', 'z', 'x'],
    'd': ['s', 'f', 'e', 'r', 'x', 'c'],
    'f': ['d', 'g', 'r', 't', 'c', 'v'],
    'g': ['f', 'h', 't', 'y', 'v', 'b'],
    'h': ['g', 'j', 'y', 'u', 'b', 'n'],
    'j': ['h', 'k', 'u', 'i', 'n', 'm'],
    'k': ['j', 'l', 'i', 'o', 'm', ','],
    'l': ['k', ';', 'o', 'p', ',', '.'],
    ';': ['l', "'", 'p', '[', '.', '/'],
    "'": [';', '\\', '[', ']', '/'],
    '\\': ["'", ']'],

    'z': ['x', 'a', 's'],
    'x': ['z', 'c', 's', 'd'],
    'c': ['x', 'v', 'd', 'f'],
    'v': ['c', 'b', 'f', 'g'],
    'b': ['v', 'n', 'g', 'h'],
    'n': ['b', 'm', 'h', 'j'],
    'm': ['n', ',', 'j', 'k'],
    ',': ['m', '.', 'k', 'l'],
    '.': [',', '/', 'l', ';'],
    '/': ['.', ';', "'"],
}
keyboard_pertub_map.update({shift_map[k]: [shift_map[vv] for vv in v] for k, v in keyboard_pertub_map.items()})
# print(keyboard_pertub_map)

alphadigits = string.ascii_letters + string.digits

data = {}
with open('data/train.txt', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        data[line] = data.get(line, 0) + 1
with open('data/test.txt', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        data[line] = data.get(line, 0) + 1
data = [(k, v) for k, v in data.items() if v > 3 and not k.startswith('ssh') and len(k.split()) > 1 and (
    any([x.startswith('-') or ('/' in x and len(x.split('/')) >= 4 and any([w.isalpha() for w in x.split('/')])) for x in k.split()])
    or any([i > 0 and i + 1 < len(k) and k[i] == ' ' and k[i - 1] in alphadigits and k[i + 1] in alphadigits for i in range(len(k))])
)]
print(len(data))

# mode = 'train'
mode = 'test'

cmds = set()
with open(f'data/{mode}.txt', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        cmds.add(line)
print(len(cmds))

with open(f'data/crafted_error_3_{mode}.txt', 'w', encoding='utf-8') as fout:
    for k, v in data:
        if k not in cmds:
            continue
        y = k
        # op = np.random.randint(3)
        xs = set()
        for op in [0] + [1] * 1 + [2] * 1 + [3] * 1:
            if op == 0:
                x = y
            else:
                x = None
                if op == 1:
                    pos = []
                    for i in range(len(k)):
                        if i > 0 and i + 1 < len(k) and k[i] == ' ' and k[i - 1] in alphadigits and k[i + 1] in alphadigits:
                            pos.append(i)
                    if len(pos) > 0:
                        i = pos[np.random.randint(len(pos))]
                        assert i > 0 and i + 1 < len(k) and k[i] == ' ' and k[i - 1] in alphadigits and k[i + 1] in alphadigits
                        x = k[:i] + k[i + 1:]
                elif op == 3:
                    t = k.split()
                    dir_pos = []
                    for i in range(len(t)):
                        if '/' in t[i]:  # is dir
                            dir_pos.append(i)
                    if len(dir_pos) > 0:
                        i = dir_pos[np.random.randint(len(dir_pos))]
                        word_pos = []
                        ttt = t[i].split('/')
                        for j in range(len(ttt)):
                            tt = ttt[j]
                            if tt.isalpha() and len(tt) >= 4:
                                word_pos.append(j)
                        if len(word_pos) > 0:
                            j = word_pos[np.random.randint(len(word_pos))]
                            tt = ttt[j]
                            i1 = np.random.randint(len(tt) - 2)
                            while True:
                                i2 = np.random.randint(len(tt) - 2)
                                if i2 != i1:
                                    break
                            if tt[i1+1] != tt[i2+1]:
                                if i1 > i2:
                                    i1, i2 = i2, i1
                                ttt[j] = tt[:i1+1] + tt[i2+1] + tt[i1+2:i2+1] + tt[i1+1] + tt[i2+2:]
                                t[i] = '/'.join(ttt)
                                x = ' '.join(t)
                elif op == 2:
                    t = k.split()
                    opt_pos = []
                    for i in range(len(t)):
                        tt = t[i]
                        if tt.startswith('-'):
                            opt_pos.append(i)
                    if len(opt_pos) > 0:
                        i = opt_pos[np.random.randint(len(opt_pos))]
                        tt = t[i]
                        assert tt.startswith('-')
                        j = np.random.randint(len(tt))
                        c = tt[j]
                        c = keyboard_pertub_map[c][np.random.randint(len(keyboard_pertub_map[c]))]
                        t[i] = tt[:j] + c + tt[j + 1:]
                        x = ' '.join(t)
            if x is not None:
                xs.add((x, op))
        for x, op in xs:
            fout.write(json.dumps({'x': x, 'y': y, 'type': op}, ensure_ascii=False) + '\n')

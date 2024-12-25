import json

def identify_path(line):
    s_line = line.strip().split()
    found = False
    pos = None
    for i, w in enumerate(s_line):
        if '.' in w or '/' in w:
            found = True
            pos = i
            break
    if found:
        w = s_line[pos]
        w = w.split('/')
        if w[0] == '':
            w = ['/' + w[1]] + w[2:]
        if w[-1] == '':
            w = w[:-1]
        cd = ['cd ' + x for x in w[:-1]]
        res = [' '.join(s_line[:pos] + [w[-1]] + s_line[pos+1:])]
        return cd + res
    else:
        return [line]

mode = 'train'
# mode = 'test'

with open(f'data/{mode}.txt', encoding='utf-8') as fin, \
        open(f'data/filtered_path_{mode}.txt', 'w', encoding='utf-8') as fout:
    for i, line in enumerate(fin):
        line = line.strip()
        s_line = line.split()
        if s_line[0] in ['cat', 'grep', 'vi', 'vim', 'head', 'tail', 'zip', 'unzip', 'zgrep', 'dgrep', 'cd', 'more', ] and len(s_line) > 1:
            # print(line, identify_path(line))
            x = identify_path(line)
            y = line
            if len(x) > 1:
                fout.write(json.dumps({
                    'x': x,
                    'y': y,
                }, ensure_ascii=False) + '\n')
            # if i == 100:
            #     break

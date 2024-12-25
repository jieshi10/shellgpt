# https://github.com/laohur/SuffixAutomaton/blob/master/SuffixAutomaton.py


from typing import List
import copy


class State:
    def __init__(self, position: int=-1, length: int=0, next=None, link: int=0) -> None:
        self.position = position  # in line
        self.length = length  # max_len
        self.link = link  # back
        if not next:
            next = {}
        self.next = next  # transation


class SuffixAutomaton:
    def __init__(self, line: List[str]) -> None:
        # self.sequence = [x for x in line if x]
        self.sequence = line
        self.last = 0
        self.size = 1
        nodes = [None for _ in range(2*len(self.sequence)+3)]
        nodes[0] = State(link=-1)
        for i, x in enumerate(self.sequence):
            nodes = self.insert(i, x, nodes)
        self.nodes = nodes[:self.size]

    def insert(self, position: int, token: str, nodes: List[State]):
        current = self.size
        self.size += 1
        # new
        nodes[current] = State(
            position=position, length=nodes[self.last].length+1)
        # 如果后缀自动机最近转移里面没有当前字符，则添加该字符，并将状态指向当前状态 继续沿着后缀连接走，进行上述操作直到到达第一个状态或者转移中有此字符
        p = self.last
        while p >= 0 and token not in nodes[p].next:
            nodes[p].next[token] = current
            p = nodes[p].link
        # 如果后缀链接走到底了，没有相同的，则后缀链接指向0状态，即空字符串
        if p == -1:
            nodes[current].link = 0
        # 如果找到上一状态的转移里有c字符,找到转移c的另一状态
        else:
            q = nodes[p].next[token]
            # 如果q状态与p状态相连，则当前状态的后缀链接指向q状态
            if nodes[p].length+1 == nodes[q].length:
                nodes[current].link = q
            # 如果不相连则开一个新状态,长度为p状态的下一个状态，后缀链接与转移指向q
            else:  # new
                clone = self.size
                self.size += 1
                nodes[clone] = State(position=position, length=nodes[p].length+1,
                                     next=copy.deepcopy(nodes[q].next), link=nodes[q].link)
                # 搜索状态p，若c转移为q，则指向新状态，并搜索后缀链接的状态重复指向新状态 直到状态转移不为q，跳出
                while p != -1 and nodes[p].next[token] == q:
                    nodes[p].next[token] = clone
                    p = nodes[p].link
                # 把当前状态与q的后缀链接指向新状态
                nodes[q].link = nodes[current].link = clone
        # 状态索引占位
        self.last = current
        return nodes
    
    def is_substring(self, query: List[str]):
        p = 0
        for x in query:
            if x in self.nodes[p].next:
                p = self.nodes[p].next[x]
            else:
                return False
        return True


if __name__ == '__main__':
    import numpy as np
    for _ in range(100):
        available_substr = [list(np.random.choice([0, 1], size=np.random.randint(3, 30))) for _ in range(100)]
        original_str = []
        if np.random.randint(2) == 1:
            original_str += list(np.random.choice([0, 1], size=np.random.randint(3, 30)))
        for s in available_substr:
            original_str += s
            if np.random.randint(2) == 1:
                original_str += list(np.random.choice([0, 1], size=np.random.randint(3, 30)))
        original_str = ''.join(map(str, original_str))
        # print(original_str)
        sa = SuffixAutomaton(original_str)
        for _ in range(1000):
            if np.random.randint(2) == 1:
                q = available_substr[np.random.randint(len(available_substr))]
            else:
                q = list(np.random.choice([0, 1], size=np.random.randint(3, 30)))
            q = ''.join(map(str, q))
            assert sa.is_substring(q) == (q in original_str)
            # print(q, sa.is_substring(q), q in original_str)
    print("OK")

import pickle
from gensim.models import KeyedVectors
from random import shuffle
import crash
import re
import numpy as np
import jieba

length = 150
w2v = KeyedVectors.load_word2vec_format('./cn.skipgram.bin', binary=True, unicode_errors="ignore")

with open('v1_mt_0.7.pkl', 'rb') as f:
    p_d = pickle.load(f)
with open('v1_lt_-0.7.pkl', 'rb') as f:
    n_d = pickle.load(f)
with open('v1_other.pkl', 'rb') as f:
    other = pickle.load(f)
x = []
for i in p_d + n_d + other:
    objs = i.newslist[:4] + i.announcelist[:4] + i.researchlist[:4]
    docs = []
    for o in objs:
        if hasattr(o, 'content'):
            docs.append(o.content)
    shuffle(docs)
    doc = ''.join(docs)
    doc = re.sub(r'\s+', '', doc)
    dv = np.zeros((length, 300))
    row_ind = 0
    for sent in re.split(r'[。；？！]', doc):
        if row_ind > length - 1:
            break
        sv = []
        wds = list(jieba.cut(sent))
        if len(wds) < 5:
            continue
        else:
            for wd in wds:
                if wd in w2v.vocab:
                    sv.append(w2v.wv[wd])
        if sv:
            dv[row_ind] = np.mean(np.array(sv), 0)
            row_ind += 1
        else:
            print('something wrong')
    x.append(dv)
y = [1] * len(p_d) + [-1] * len(n_d) + [0] * len(other)

with open('cnnTrainsetall.pkl', 'wb') as f:
    pickle.dump({'x': x, 'y': y}, f)

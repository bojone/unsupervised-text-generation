#! -*- coding: utf-8 -*-
# MCMC造句
# 参考：https://kexue.fm/archives/8194

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout
from tqdm import tqdm

config_path = '/root/kg/bert/chinese_nezha_gpt_L-12_H-768_A-12/config.json'
checkpoint_path = '/root/kg/bert/chinese_nezha_gpt_L-12_H-768_A-12/gpt.ckpt'
dict_path = '/root/kg/bert/chinese_nezha_gpt_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True, token_end=u'。')  # 建立分词器

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    segment_vocab_size=0,  # 去掉segmeng_ids输入
    application='lm',
)  # 建立模型，加载权重


def LogProba(token_ids, gamma=0.8):
    """用语言模型算一个句子的概率对数
    gamma：0～1之间，值越小将会鼓励生成越长的句子。
    """
    probas = model.predict(np.array([token_ids[:-1]]))[0]
    log_probas = np.log(probas + 1e-8)
    log_probas = [log_probas[i, j] for i, j in enumerate(token_ids[1:])]
    return sum(log_probas) * gamma


def ReNormProba(batch_token_ids):
    """每个句子算自己的概率值，然后重新归一化
    """
    batch_token_ids = np.array(batch_token_ids)
    probas = model.predict(batch_token_ids[:, :-1])
    log_probas = np.log(probas + 1e-8)
    log_probas = np.array([
        sum([log_probas[k, i, j]
             for i, j in enumerate(token_ids[1:])])
        for k, token_ids in enumerate(batch_token_ids)
    ])
    log_probas -= log_probas.max()
    probas = np.exp(log_probas)
    return probas / probas.sum()


def TopCandidates(token_ids, i, topn=64):
    """用（单向）语言模型给出第i个位置的topn个候选token
    """
    token_ids_i = token_ids[i]
    token_ids = np.array([token_ids[:i]])
    probas = model.predict(token_ids)[0, -1]
    ids = list(probas.argsort()[::-1][:topn])
    if token_ids_i in ids:
        ids.remove(token_ids_i)
    else:
        ids = ids[:-1]
    return [token_ids_i] + ids  # 将输入token放在第一位，方便调用


def replace(token_ids, i, value):
    """将token_ids的第i个值替换为value
    """
    return token_ids[:i] + [value] + token_ids[i + 1:]


def insert(token_ids, i):
    """在token_ids的第i个位置插入一个mask
    """
    return token_ids[:i] + [tokenizer._token_mask_id] + token_ids[i:]


def delete(token_ids, i):
    """删除token_ids的第i个元素
    """
    return token_ids[:i] + token_ids[i + 1:]


def track_of(results):
    print(u'输入词语：%s' % u'、'.join(words))
    print(u'开始状态：%s' % results[0])
    last = results[0]
    for r in results[1:]:
        if r[2] != last:
            print(u'第%s步，执行%s, 输出：%s' % tuple(r))
            last = r[2]


words = [u'广州', u'美食', u'开幕']
# words = [u'科学', u'空间']
# words = [u'我们', u'数学']
token_ids = [tokenizer._token_start_id]
masked_idxs, masked_ends = [], []
for w in words:
    ids = tokenizer.encode(w)[0][1:-1]
    masked_idxs.extend(range(len(token_ids), len(token_ids) + len(ids)))
    masked_ends.append(len(token_ids) + len(ids) - 1)
    token_ids.extend(ids)

token_ids.append(tokenizer._token_end_id)
results = [tokenizer.decode(token_ids)]

steps = 128
ops = ['replace', 'insert', 'delete']

for _ in tqdm(range(steps), desc='Sampling'):
    # 选择操作
    op = np.random.choice(ops)
    # 选择位置
    idxs = [i for i in range(1, len(token_ids) - 1) if i not in masked_idxs]
    if op == 'insert':
        idxs = [0] + list(set(idxs + masked_ends))
    if not idxs:
        continue
    i = np.random.choice(idxs)
    # 执行操作
    if op == 'replace':
        ids = TopCandidates(token_ids, i)
        batch_token_ids = [replace(token_ids, i, j) for j in ids]
        replace_probas = ReNormProba(batch_token_ids)
        k = np.random.choice(len(ids), p=replace_probas)
        new_token_ids = replace(token_ids, i, ids[k])
        # 计算接受率
        log_proba = LogProba(token_ids)
        new_log_proba = LogProba(new_token_ids)
        alpha = np.exp(new_log_proba - log_proba)
        alpha = alpha * replace_probas[0] / replace_probas[k]
        # 采样决定
        if np.random.random() < alpha:
            token_ids = new_token_ids
    elif op == 'insert':
        i += 1
        new_token_ids = insert(token_ids, i)
        ids = TopCandidates(new_token_ids, i)
        batch_token_ids = [replace(new_token_ids, i, j) for j in ids]
        replace_probas = ReNormProba(batch_token_ids)
        k = np.random.choice(len(ids), p=replace_probas)
        new_token_ids = replace(new_token_ids, i, ids[k])
        # 计算接受率
        log_proba = LogProba(token_ids)
        new_log_proba = LogProba(new_token_ids)
        alpha = np.exp(new_log_proba - log_proba) / replace_probas[k]
        # 采样决定
        if np.random.random() < alpha:
            token_ids = new_token_ids
            masked_idxs = [j + int(j >= i) for j in masked_idxs]
            masked_ends = [j + int(j >= i) for j in masked_ends]
    else:
        ids = TopCandidates(token_ids, i)
        k = ids.index(token_ids[i])
        batch_token_ids = [replace(token_ids, i, j) for j in ids]
        replace_probas = ReNormProba(batch_token_ids)
        new_token_ids = delete(token_ids, i)
        # 计算接受率
        log_proba = LogProba(token_ids)
        new_log_proba = LogProba(new_token_ids)
        alpha = np.exp(new_log_proba - log_proba) * replace_probas[k]
        # 采样决定
        if np.random.random() < alpha:
            token_ids = new_token_ids
            masked_idxs = [j - int(j >= i) for j in masked_idxs]
            masked_ends = [j - int(j >= i) for j in masked_ends]
    results.append((_, op, tokenizer.decode(token_ids)))


track_of(results)

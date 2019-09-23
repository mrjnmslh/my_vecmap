#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: Eval_G.py
@time: 2018/8/23 12:51
"""

# coding:utf-8
from myembedding import WordEmbeddings
import torch
from torch.autograd import Variable
import argparse
import copy
import os
import numpy as np
from timeit import default_timer as timer
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

torch.cuda.set_device(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for Unsupervised Bilingual Lexicon Induction using GANs')
    parser.add_argument("--src_lang", type=str, default='en')
    parser.add_argument("--src_emb", type=str)
    parser.add_argument("--tgt_lang", type=str, default='fr')
    parser.add_argument("--tgt_emb", type=str)
    parser.add_argument("--exp_name", type=str, default='mono')

    return parser.parse_args()

def _get_eval_params(params):
    params = copy.deepcopy(params)
    params.ks = [1, 5, 10]
    params.methods = ['nn', 'csls']
    params.models = ['procrustes', 'adv']
    params.refine = ['without-ref', 'with-ref']
    return params


def plot(file,word_list,word_list2=None,lang='',exp_name='mono'):

    we1 = WordEmbeddings()
    we1.load_from_word2vec_new(file, full_vocab=False)

    src = we1.vectors[:20000]
    tsne = TSNE(n_components=2)
    tmp1 = tsne.fit_transform(src)
    trainX = []
    # tmp = list(list(we1.word2id.items())[0])[0]
    # print(type(tmp),tmp)
    # print(we1.word2id)
    for word in word_list:
        # print(type(word))
        trainX.append(tmp1[we1.word2id[str(word)]])
    X = np.array(trainX)
    fig = plt.figure(figsize=(12, 10), dpi=175)
    plt.xlim(X[:, 0].min() * 1.1, X[:, 0].max() + 2.5)
    plt.ylim(X[:, 1].min() * 1.1, X[:, 1].max() + 1.0)
    plt.scatter(X[:, 0], X[:, 1], c='r')
    for i in range(len(X[:, 0])):
        plt.text(X[:, 0][i], X[:, 1][i], str(word_list2[i] if word_list2 else word_list[i]), color='r', wrap=True)
    plt.show()
    plt.savefig('{}-{}.png'.format(exp_name,lang))

def main():
    params = parse_arguments()
    words_list = ['year','american','time','life','world','people','person','man','discussion','article']
    words_list2 = ['年','美国','时间','生活','世界','人们','人','男人','讨论','文章']
    words_list3 = ['nian','meiguo','shijian','shenghuo','shijie','renmem','ren','nanren','taolun','wenzhang']


    plot(params.tgt_emb,words_list2,words_list3,lang=params.tgt_lang,exp_name=params.exp_name)
    plot(params.src_emb, words_list, lang=params.src_lang,exp_name=params.exp_name)


if __name__ == '__main__':
    main()

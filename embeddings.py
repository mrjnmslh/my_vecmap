# coding:utf-8
# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cupy_utils import *

import numpy as np


def read(file, threshold=0, vocabulary=None, dtype='float'):
    print('Loading pre-trained embeddings from',file.name)
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words,cnt = [],0
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    while cnt < count:
        line = file.readline()
        if not line:
            break   
        word, vec = line.strip().split(' ', 1)
        if len(vec.split())==dim and word not in words:
            if vocabulary is None:
                words.append(word)
                matrix[cnt] = np.fromstring(vec, sep=' ', dtype=dtype)
            elif word in vocabulary:
                words.append(word)
                matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
            cnt+=1
        else:
            pass
            # print('Invalid demension {}'.format(len(vec.split())))
            # print('Skip invalid embedding {}'.format(word))
    print('Loaded {} embeddings'.format(cnt))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))

def read_new(file, threshold=0, vocabulary=None, dtype='float'):
    print('Loading pre-trained embeddings from',file.name)
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words,cnt = [],0
    matrix = []
    while cnt < count:
        line = file.readline()
        if not line:
            break   
        word, vec = line.strip().split(' ', 1)
        if len(vec.split())==dim and word not in words:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
            cnt+=1
        else:
            pass
            # print('Invalid demension {}'.format(len(vec.split())))
            # print('Skip invalid embedding {}'.format(word))
    print('Loaded {} embeddings'.format(cnt))
    return (words, np.array(matrix, dtype=dtype))

def write(words, matrix, file):
    m = asnumpy(matrix)
    print('%d %d' % m.shape, file=file)
    for i in range(len(words)):
        print(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]), file=file)


def length_normalize(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, xp.newaxis]


def mean_center(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=0)
    matrix -= avg


def length_normalize_dimensionwise(matrix):
    xp = get_array_module(matrix)
    norms = xp.sqrt(xp.sum(matrix**2, axis=0))
    norms[norms == 0] = 1
    matrix /= norms


def mean_center_embeddingwise(matrix):
    xp = get_array_module(matrix)
    avg = xp.mean(matrix, axis=1)
    matrix -= avg[:, xp.newaxis]


def normalize(matrix, actions):
    for action in actions:
        if action == 'unit':
            length_normalize(matrix)
        elif action == 'center':
            mean_center(matrix)
        elif action == 'unitdim':
            length_normalize_dimensionwise(matrix)
        elif action == 'centeremb':
            mean_center_embeddingwise(matrix)

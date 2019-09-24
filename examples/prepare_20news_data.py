"""functions to process the 20 news group data_set
Details about the data can be found here: http://qwone.com/~jason/20Newsgroups/
a processed version of the 20news-bydate data set is used: 20news-bydate-matlab.tgz
    train.data
    train.label
    train.map
    test.data
    test.label
    test.map
"""

import sys, os
sys.path.append("/".join(os.path.split(os.path.realpath(__file__))[:-1]) + "/../utils/")

from gensim import corpora
from collections import OrderedDict
from prepare_text_data import filter_tokens, build_new_corpus, save
import os
import numpy as np
import pandas as pd

def read_data(infile):
    """read the .data files formatted with "docIdx wordIdx count",
    construct corresponding BOW corpus
    """
    bow = OrderedDict()
    with open(infile, "r") as fin:
        for line in fin:
            docid, wordid, cnt = [int(item) for item in line.split()]
            wordid -= 1
            if docid not in bow:
                bow[docid] = []
            bow[docid].append((wordid, cnt))
    return list(bow.values())

def prepare(train_file, otherfiles=[], mindocfreq=5, maxdocfreq=0.9, keepNword=None,
    filterby="df", vocabulary_file=None, outfolder="./"):
    bow_train = read_data(train_file)
    bow_other = [read_data(item) for item in otherfiles]
    id2word = None
    if vocabulary_file:
        with open(vocabulary_file, "r") as fin:
            id2word = {i:item.strip() for i, item in enumerate(fin)}
    dictionary  = corpora.Dictionary.from_corpus(bow_train, id2word)
    cfs = {}
    for doc in bow_train:
        for id, cnt in doc:
            cfs[id] = cfs.get(id, 0) + cnt
    dictionary, convert_map = filter_tokens(dictionary, mindocfreq, maxdocfreq, keepNword, filterby, cfs)

    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    for corpus, filename in zip([bow_train] + bow_other, [train_file] + otherfiles):
        #raise
        corpus, tfidfcorpus, numpy_matrix, numpy_matrix_tfidf = build_new_corpus(dictionary,
            corpus, convert_map)
        desc = os.path.splitext(os.path.split(filename)[-1])[0]
        print(filename, desc)
        save(corpus, tfidfcorpus, numpy_matrix, numpy_matrix_tfidf, None, outfolder, desc)
    dictionary.save(os.path.join(outfolder, "dictionary"))

def prepare_label(*infiles):# so that the label starts from 0
    for infile in infiles:
        data = pd.read_csv(infile, header=None)
        data[data.columns[0]] -= 1
        prefix, ext = os.path.splitext(infile)
        data.to_csv(prefix + "_zerobased" + ext, index=False, header=False)

if __name__ == "__main__":
    prepare("./data/20newsgroup/20news-bydate-matlab/train.data",
    ["./data/20newsgroup/20news-bydate-matlab/test.data"],
    keepNword=10000, filterby="tfidf",
    vocabulary_file="./data/20newsgroup/20news-bydate-matlab/vocabulary.txt",
    outfolder="./data/20newsgroup/")
    prepare_label("./data/20newsgroup/train.label",
        "./data/20newsgroup/test.label")

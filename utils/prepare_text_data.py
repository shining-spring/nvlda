"""A couple of util functions for manipulating corpus
use gensim for text vectorization etc.
"""
from gensim import corpora, matutils, models
import os
import numpy as np

def filter_tokens(dictionary, mindocfreq=5, maxdocfreq=0.9, keepNword=None,
    filterby="df", cfs=None):
    """Filter tokens in dictionary using standards as minimal/max doc freq, sort
    tokens by doc freq or tfidf and keep only the top N words
    parameters:
        dictionary: gensim.corpora.Dictionary instance
        mindocfreq: minimal doc freq for the tokens, actual number or percentage
        maxdocfreq: maximal doc freq for the tokens, actual number or percentage
        keepNword: number of tokens to keep, if None, then all tokens after filtering
            by min and max docfreq
        filterby: if keepNword is set, then sort the tokens by doc freq or tfidf
        cfs: corpus frequency of tokens, used to calculate tfidf
    returns:
        dictionary: filtered dictionary
        convert_map: map of tokenids between old and new dictionary
    """
    token2id = dictionary.token2id.copy()
    if mindocfreq < 1:
        mindocfreq = dictionary.num_docs * mindocfreq
    if maxdocfreq < 1:
        maxdocfreq = dictionary.num_docs * maxdocfreq
    toremoveid = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq <= mindocfreq or docfreq >= maxdocfreq]

    if keepNword:#
        print("keep top %i words ranked by %s"%(keepNword, filterby))
        if filterby == "df": #filter by df the top N words
            toremoveid += [item[0] for item in sorted(dictionary.dfs.iteritems(),
                key=lambda s:s[1], reverse=True)[keepNword:]]
        elif filterby == "tfidf": # filter by tfidf the top N words
            if cfs is None and len(dictionary.cfs) == 0:
                print("either cfs has to be provided or cfs of dictionary has to be populated")
                return
            if cfs is None:
                cfs = dictionary.cfs
            alltfidf = [(key, item * models.tfidfmodel.df2idf(dictionary.dfs[key], dictionary.num_docs))
                for key, item in cfs.items()]
            toremoveid += [item[0] for item in sorted(alltfidf, key=lambda s:s[1], reverse=True)[keepNword:]]
    dictionary.filter_tokens(toremoveid)
    dictionary.compactify()

    convert_map = {token2id[key] : item for key, item in dictionary.token2id.items()}
    return dictionary, convert_map

def build_new_corpus(dictionary, corpus, convert_map=None):
    """build new corpus based on new dictionary and then tfidf corpus
    parameters:
        dictionary: gensim.corpora.Dictionary instance
        corpus: bow corpus to be processed
        convert_map: dictionary to map old token ids into new token ids
    returns:
        corpus: new corpus according to new dictionary
        tfidfcorpus: tfidf corpus
        numpy_matrix: dense matrix version of corpus
        numpy_matrix_tfidf: dense matrix version of tfidf corpus
    """
    #raise
    if convert_map:
        corpus = [[(convert_map[id], cnt) for id, cnt in doc if id in convert_map] for doc in corpus]

    number_of_corpus_features = len(dictionary)
    print(len(corpus), number_of_corpus_features)
    numpy_matrix = matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)

    tfidf = models.TfidfModel(dictionary=dictionary)
    tfidfcorpus = tfidf[corpus]
    numpy_matrix_tfidf = matutils.corpus2dense(tfidfcorpus, num_terms=number_of_corpus_features)
    return corpus, tfidfcorpus, numpy_matrix.T, numpy_matrix_tfidf.T # so that the matrix is of the dimension nsamples x nfeatures

def save(corpus, tfidfcorpus, numpy_matrix, numpy_matrix_tfidf, meta, outfolder, desc):
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    corpora.MmCorpus.serialize(os.path.join(outfolder, "corpus_bow_%s.mm"%desc), corpus)
    corpora.MmCorpus.serialize(os.path.join(outfolder, "corpus_tfidf_%s.mm"%desc), tfidfcorpus)

    np.save(os.path.join(outfolder, "corpus_bow_%s.npy"%desc), numpy_matrix)
    np.save(os.path.join(outfolder, "corpus_tfidf_%s.npy"%desc), numpy_matrix_tfidf)
    if meta is not None:
        meta.to_csv(os.path.join(outfolder, "corpus_%s.meta"%desc), index=False)

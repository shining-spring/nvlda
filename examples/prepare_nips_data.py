"""functions to process the nips papers data_set
Details about the data can be found here: https://www.kaggle.com/benhamner/nips-papers
there are basically three files:
authors.csv: with id, name columns contains a list of all authors for nips papers from 1987-2017
papers_authors.csv, with id, paper_id, author_id maps each papers to a list of authors
papers.csv: with id, year, title, event_type, pdf_name, abstract, paper_text contains
    text and meta information about the paper. Full text of paper is in the text column.
"""
import pandas as pd
import numpy as np
from gensim import corpora, matutils
import sys, os
sys.path.append("/".join(os.path.split(os.path.realpath(__file__))[:-1]) + "/../utils/")
from prepare_text_data import filter_tokens, build_new_corpus, save
import spacy
import random


def process_paper_author(infile, author_file, outfolder, minpapers=5, test=True):
    """
    prepare paper_authors.csv file, generate onehot distribution of authors for each paper,
    filter author list by total number of papers published.
    parameters:
        infile: path to paper_authors.csv
        author_file: path to authors.csv
        outfolder: path to store output files
        minpapers: minimal number of papers for the author to be included
        test: if True only first 50 rows in paper_authors.csv processed
    """
    nrows = 50 if test else None
    data = pd.read_csv(infile, nrows=nrows).groupby(["paper_id", "author_id"]).size().unstack()#.reset_index()
    print(data.shape)
    # data.to_csv(os.path.join(outfolder, os.path.splitext(os.path.split(infile)[-1])[0] + "_authors_onehot.csv"), index=False)

    papers_by_author = data.sum(axis=0)
    tokeep = papers_by_author[papers_by_author >= minpapers].index
    data = data[tokeep]
    print(data.shape)

    authors = pd.read_csv(author_file)

    authors_t = pd.DataFrame({"author_id" : data.columns.values})#[1:]
    authors_t["id"] = authors_t.index
    authors_t = pd.merge(authors_t, authors.rename(columns={"id" : "author_id"}), on="author_id", how="left")
    authors_t.to_csv(os.path.join(outfolder, "author_ids.map"), index=False)
    return data


def parse_text(infile, nlp=None, mindocfreq=5, maxdocfreq=0.9, keepNword=None,
    filterby="df", paper_author_file="", author_file="", minpapers=5, test=True,
    outfolder=".", **kwargs):
    """parse papers.csv file, process text in text column with spacy, filter
    tokens by min/max doc freq and keep only keepNword tokens, and output
    resulting corpus and dictionary. process papers_authors.csv file, keep only authors
    with at least minpapers articles, use papers with at least one author in the kept
    list for train, the rest for test.
    parameters:
        infile: path to the papers.csv file
        nlp: spacy nlp instance, if None, will be initiated
        mindocfreq: minimal doc freq for the tokens, actual number or percentage
        maxdocfreq: maximal doc freq for the tokens, actual number or percentage
        keepNword: number of tokens to keep, if None, then all tokens after filtering
            by min and max docfreq
        filterby: if keepNword is set, then sort the tokens by doc freq or tfidf
        paper_author_file: path to paper_authors.csv
        author_file: path to authors.csv
        minpapers: minimal number of papers for the author to be included
        test: if True will only process the first 50 rows in the input file
        outfolder: path to the output folder to save preprocessed corpus
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
    nrows = 500 if test else None
    data = pd.read_csv(infile, nrows=nrows)
    paper_author = process_paper_author(paper_author_file, author_file,
        outfolder, minpapers, test=False)

    paper_author = pd.merge(data[["id"]], paper_author, left_on="id",
        right_index=True, how="left").reset_index(drop=True).drop(["id"], axis=1)

    idx_train = paper_author[paper_author.sum(axis=1) > 0].index
    idx_test = paper_author[paper_author.sum(axis=1) == 0].index
    paper_author.loc[idx_train].to_csv(os.path.join(outfolder, "corpus_train_authors_onehot.csv"),
        index=False, header=False)

    paper_author.loc[idx_test].to_csv(os.path.join(outfolder, "corpus_test_authors_onehot.csv"),
        index=False, header=False)

    documents = [[token.lemma_ for token in doc if (not token.is_stop
        and token.is_alpha and len(token.lemma_) > 1)]#
        for doc in nlp.pipe(data["paper_text"].str.lower().tolist())]


    data_train, data_test = data.loc[idx_train], data.loc[idx_test]
    documents_train = [documents[i] for i in idx_train]
    documents_test = [documents[i] for i in idx_test]
    # raise
    dictionary = corpora.Dictionary(documents_train)
    documents_train = [dictionary.doc2bow(doc) for doc in documents_train]
    cfs = {}
    for doc in documents_train:
        for id, cnt in doc:
            cfs[id] = cfs.get(id, 0) + cnt
    dictionary, convert_map = filter_tokens(dictionary, mindocfreq, maxdocfreq,
        keepNword, filterby, cfs)
    corpus, tfidfcorpus, numpy_matrix, numpy_matrix_tfidf = \
        build_new_corpus(dictionary, documents_train, convert_map)
    save(corpus, tfidfcorpus, numpy_matrix, numpy_matrix_tfidf,
        data_train.drop("paper_text", axis=1), outfolder, "train")

    documents_test = [dictionary.doc2bow(doc) for doc in documents_test]
    corpus_test, tfidfcorpus_test, numpy_matrix_test, numpy_matrix_tfidf_test = \
        build_new_corpus(dictionary, documents_test, None)
    save(corpus_test, tfidfcorpus_test, numpy_matrix_test, numpy_matrix_tfidf_test,
        data_test.drop("paper_text", axis=1), outfolder, "test")

    dictionary.save(os.path.join(outfolder, "dictionary"))



if __name__ == "__main__":
    parse_text("./data/nips/papers.csv",
        None, outfolder="./data/nips", paper_author_file="./data/nips/paper_authors.csv",
        author_file="./data/nips/authors.csv", minpapers=5, test=False,
        keepNword=10000, filterby="tfidf", mindocfreq=5, maxdocfreq=0.9)

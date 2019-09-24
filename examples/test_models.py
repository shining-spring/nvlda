import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append("/".join(os.path.split(os.path.realpath(__file__))[:-1]) + "/../../")
from nvlda.models import build_model
from nvlda.utils import read_config

if len(sys.argv) == 3:
    configfile = sys.argv[1]
    config = sys.argv[2]
else:
    configfile = "test_models.conf"
    config = sys.argv[1]

for thisconfig in config.split(","):
    print(configfile, thisconfig)
    settings = read_config(configfile, section=thisconfig)
    wfolder = getattr(settings, "wfolder",
        "./data/")
    corpus_file = getattr(settings, "corpus_file", "corpus_bow.mm")
    dictionary_file = getattr(settings, "dictionary_file", "dictionary")
    meta_file = getattr(settings, "meta_file", "corpus_meta")
    num_topics = getattr(settings, "num_topics", 50)
    type_model = getattr(settings, "type_model", "lda")
    modelkwargs = getattr(settings, "modelkwargs", {})
    train_kwargs = getattr(settings, "train_kwargs", {})
    input_x_file = getattr(settings, "input_x_file", None)
    input_x_categorical = getattr(settings, "input_x_categorical", True)
    input_x_label_file = getattr(settings, "input_x_label_file", None)
    input_x_label_colname = getattr(settings, "input_x_label_colname", "name")
    test_file = getattr(settings, "test_file", None)
    test_meta_file = getattr(settings, "test_meta_file", None)
    test_unknown_value = getattr(settings, "test_unknown_value", 1)
    x_num_classes = modelkwargs.get("x_num_classes", None)

    if x_num_classes and input_x_file is None:
        print("input_x_file must be specified, pass this config")
        continue

    dictionary = gensim.corpora.Dictionary()
    dictionary = dictionary.load(os.path.join(wfolder, dictionary_file))
    metas = pd.read_csv(os.path.join(wfolder, meta_file))

    if type_model == "lda":
        corpus = gensim.corpora.MmCorpus(os.path.join(wfolder, corpus_file))
        print("no. docs: %i"%len(corpus))
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary.id2token,
            num_topics=num_topics, **modelkwargs)
        model.save(os.path.join(wfolder, "%s.pkl"%thisconfig))
        gamma, _ = model.inference(corpus)
        topic_dist = gamma / gamma.sum(axis=1)[:, None]
        topic_dist = pd.DataFrame(topic_dist)

        topic_word_dist = pd.DataFrame(model.get_topics().T)
    else:#type_model == "nvdm":
        numpy_matrix = np.load(os.path.join(wfolder, corpus_file))
        model = build_model(type_model=type_model, n_topic=num_topics,
            vocab_size=len(dictionary), **modelkwargs)
        if input_x_file:
            input_x = pd.read_csv(os.path.join(wfolder, input_x_file), header=None)
            if input_x_categorical:
                majority_class = input_x[input_x.columns[-1]].value_counts().index[0]
                input_x_tmp = input_x[input_x.columns[-1]].fillna(majority_class).astype(np.int32).values#[:, None]
                input_x = np.zeros((input_x_tmp.shape[0], x_num_classes), dtype=np.float32)
                input_x[np.arange(input_x.shape[0]), input_x_tmp] = 1
            else:
                input_x = input_x.fillna(0)
                input_x.loc[input_x.sum(axis=1) == 0] += test_unknown_value
                input_x = input_x.values.astype(np.float32)
            print(numpy_matrix.shape, input_x.shape)
            model.fit(numpy_matrix, input_x, callbacks=[tf.keras.callbacks.TerminateOnNaN()], **train_kwargs)
            topic_dist = pd.DataFrame(model.encode(numpy_matrix, input_x))
            prior_dist = pd.DataFrame(model.get_prior_topic_dist())
            if input_x_label_file:
                input_x_label = pd.read_csv(input_x_label_file)[input_x_label_colname].tolist()
                prior_dist.columns = input_x_label
            prior_dist.to_csv(os.path.join(wfolder, "prior_topic_dist_%s.csv"%thisconfig), index=False)
        else:
            model.fit(numpy_matrix, callbacks=[tf.keras.callbacks.TerminateOnNaN()], **train_kwargs)
            topic_dist = pd.DataFrame(model.encode(numpy_matrix))

        if test_file:
            numpy_matrix_test = np.load(os.path.join(wfolder, test_file))
            if input_x_file:
                input_x_test = np.ones((numpy_matrix_test.shape[0], x_num_classes), dtype=np.float32) * test_unknown_value
                topic_dist_test = pd.DataFrame(model.encode(numpy_matrix_test, input_x_test))
            else:
                topic_dist_test = pd.DataFrame(model.encode(numpy_matrix_test))
            topic_dist_test["top_topic"] = topic_dist_test.idxmax(axis=1)
            test_meta = pd.read_csv(os.path.join(wfolder, test_meta_file))
            topic_dist_test = pd.concat([topic_dist_test, test_meta], axis=1)
            topic_dist_test.to_csv(os.path.join(wfolder, "document_topic_dist_test_%s.csv"%thisconfig), index=False)

        topic_word_dist = pd.DataFrame(model.get_topic_word_dist().T)

    topic_dist["top_topic"] = topic_dist.idxmax(axis=1)
    topic_dist = pd.concat([topic_dist, metas], axis=1)
    topic_dist.to_csv(os.path.join(wfolder, "document_topic_dist_%s.csv"%thisconfig), index=False)
    topic_word_dist["term"] = [dictionary.id2token[i] for i in range(topic_word_dist.shape[0])]
    topic_word_dist[["term"] + [i for i in range(num_topics)]].to_csv(os.path.join(wfolder,
        "topic_word_dist_%s.csv"%thisconfig), index=False)

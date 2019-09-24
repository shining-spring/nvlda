# Neural variational topic modeling

This is an implementation of neural variational document models using Tensorflow 2.0 and Keras included with it.

The following models are implemented in models/nvlda_models.py:
- NVDM model the same as https://github.com/ysmiao/nvdm as in the paper https://arxiv.org/abs/1511.06038
- neural variational LDA and ProdLDA as https://github.com/akashgit/autoencoding_vi_for_topic_models has in the paper ttps://arxiv.org/abs/1703.01488
- neural variational author topic model as described in https://arxiv.org/abs/1207.4169, this is probably the first implementation of this model with neural variational framework. The original paper used Gibbs sampling for model parameter estimation, while in this work an inference network is trained. With this model, in addition to the topic-word distribution, we can also get author-topic distribution, i.e. a representation for authors are learned, which can be used in downstream applications to e.g. cluster authors, find authors writing similar topics. The concept of authors can also be extended to other meta information of articles, e.g. categories of the article etc. With this model, we can provide the model some prior knowledge about the data, which has been difficult for traditional LDA models.
- neural variational meta topic model, this is an extension to ProdLDA, where in addition to generate text, the topics are also supposed to generate other meta data. The output is actually similar to author topic model, where we learn a distribution of meta information given a topic, while in author topic model we learn a distribution of topics given the meta information.

In examples/ are some example scripts running the models on
- 20 News Group data http://qwone.com/~jason/20Newsgroups/
- NIPS papers data https://www.kaggle.com/benhamner/nips-papers

The resulting topic word distributions, author topic distributions are explored in the accompanying notebooks in the same folder.

Dependencies:
tensorflow 2.0, numpy, pandas, gensim (for text preprocessing), spacy (for text preprocessing)

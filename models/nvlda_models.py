"""
Some model definitions for neural variational LDA, based on
Neural Variational Inference for Text Processing, Yishu Miao, Lei Yu, Phil Blunsom, 2016
Autoencoding Variational Inference For Topic Models, Akash Srivastava, Charles Sutton, 2017
The Author-Topic Model for Authors and Documents, Michal Rosen-Zvi, Thomas Griffiths, Mark Steyvers, Padhraic Smyth, 2012
"""

import tensorflow as tf
import numpy as np
import math
def kld(inferred, prior, dist1, dist2, mask=None):
    """calculate the kl divergence between two distributions, currently only
    Gaussian-Gaussian, Multinomial-Multinomial implemented
    Params:
        inferred: tuple of mean and log sigma of inferred distribution, both in shape [batch, n_topic]
        prior: tuple of mean and log sigma of prior distribution, both in shape [batch, n_topic]
        dist1: type of distribution for inferred distribution
        dist2: type of distribution for prior distribution
        mask: mask of shape [batch, n_topic], only sum over certain topics
    returns:
        kld: the kl divergence in shape [batch]
    """
    print(dist1, dist2)
    if dist1 == "Gaussian":
        if dist2 == "Gaussian":
            h_mean, h_logsigm = inferred # both in [batch, n_topic] shape
            p_mean, p_logsigm = prior # both in [batch, n_topic] shape
            kld = (1. - tf.square((h_mean - p_mean) / (tf.exp(p_logsigm) + 1e-5))
                + 2. * h_logsigm - 2. * p_logsigm - tf.exp(2. * h_logsigm) / (tf.exp(2. * p_logsigm) + 1e-5))
            if mask is not None:
                mask = tf.cast(mask, kld.dtype)
                kld = kld * mask * tf.cast(tf.reduce_sum(mask, axis=1, keepdims=True) > 1, kld.dtype)
            kld = -0.5 * tf.reduce_sum(kld, 1)
    elif dist1 == "Multinomial":
        if dist2 == "Multinomial": # both inferred and prior in [batch, n_topic] shape
            h_mean, _ = inferred
            p_mean, _ = prior
            # h_mean, p_mean = tf.nn.softmax(h_mean), tf.nn.softmax(p_mean)
            kld = tf.reduce_sum(h_mean * (tf.math.log(h_mean + 0.00001) - tf.math.log(p_mean + 0.00001)), 1)
            #ref https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    return kld #[batch] shape

def logdirichlet_tf(concentration, x):
    """calculate the log probability of simplex x given concentration under dirichlet
    distribution
    Params:
        concentration: ? * k tensor
        x: ? * k tensor
    """
    return tf.reduce_sum(tf.math.xlogy(concentration - 1., x), axis=-1) - tf.math.lbeta(concentration)

def build_model(type_model, **kwargs):
    print(kwargs)
    if type_model == "NVDM":
        return NVDM(**kwargs)
    elif type_model == "LDA":
        return LDA(**kwargs)
    elif type_model == "ProdLDA":
        return ProdLDA(**kwargs)
    elif type_model == "ProdLDAwMeta":
        return ProdLDAwMeta(**kwargs)
    elif type_model == "AuthorTopicModel":
        return AuthorTopicModel(**kwargs)

class BaseNVDM(object):
    """
    The base NVDM class
    """
    def _init(self, vocab_size, n_topic, supervised=False, input_x_size=1,
        variational_dist="Gaussian", prior_dist="Gaussian", **kwargs):
        self.vocab_size = vocab_size
        self.n_topic = n_topic
        self.supervised = supervised # whether to include supervision
        self.input_x_size = input_x_size # for supervised model, the size of the supervision
        self.variational_dist = variational_dist
        self.prior_dist = prior_dist
        self.kwargs = kwargs # this will be passed to optimizer
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

    def __init__(self, vocab_size, n_topic, variational_dist="Gaussian",
        prior_dist="Gaussian"):
        self._init(vocab_size, n_topic, variational_dist, prior_dist)
        self.build()

    def get_recon_loss(self):
        self.recon_loss = -tf.reduce_sum(self.logits * self.inputs, 1)

    def get_kld(self):
        self.kld = kld((self.h_mean, self.h_logsigm), (self.p_mean, self.p_logsigm),
            self.variational_dist, self.prior_dist)

    def build(self):
        self.inputs = tf.keras.Input(shape=(self.vocab_size,))
        if self.supervised:
            self.inputs_x = tf.keras.Input(shape=(self.x_num_classes,), dtype="float32")
        self.build_prior_net()
        self.build_inference_net()
        self.build_generative_net()
        self.get_kld()
        self.get_recon_loss()#self.recon_loss = -tf.reduce_sum(self.logits * self.inputs, 1)

        if self.supervised:
            self.model = tf.keras.Model([self.inputs, self.inputs_x], self.logits)
        else:
            self.model = tf.keras.Model(self.inputs, self.logits)
        self.model.add_loss(tf.reduce_mean(self.kld + self.recon_loss))
        #self.model.add_metric(perplexity, name='log_perplexity')
        self.model.add_metric(self.kld, name='kld', aggregation='mean')
        self.model.add_metric(self.recon_loss, name='recon_loss', aggregation='mean')
        optimizer = tf.keras.optimizers.Adam(**self.kwargs)
        self.model.compile(optimizer=optimizer)


    def build_inference_net(self):
        """Inference net should compute a h_mean and h_logsigm
        """
        raise NotImplementedError

    def build_prior_net(self):
        """Prior net should compute a p_mean and p_logsigm
        """
        raise NotImplementedError

    def build_generative_net(self):
        """generative net should compute logits for each word in the vocab based
        on h_mean and h_logsigm which is the conditional probability of words xi given
        h (h_mean and h_logsigm), and also generate topic_word_dist matrix
        """
        raise NotImplementedError

    def get_topic_word_dist(self):
        return self.topic_word_dist.numpy()

    def encode(self, *args):
        return self.encoder(args).numpy()

    def fit(self, *args, **kwargs):
        self.model.fit(args, **kwargs)

    def save(self, outfile):
        self.model.save(outfile)

    def load(self, infile):
        self.model = tf.keras.models.load_model(infile)


class NVDM(BaseNVDM):
    """NVDM model the same as https://github.com/ysmiao/nvdm as in the paper
    https://arxiv.org/abs/1511.06038
    """
    def __init__(self, vocab_size, n_topic, n_hidden, non_linearity, **kwargs):
        self._init(vocab_size, n_topic, **kwargs)
        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        if isinstance(non_linearity, str):
            print("non_linearity", non_linearity)
            self.non_linearity = eval(non_linearity)
        self.build()

    def build_inference_net(self):
        self.enc_vec = tf.keras.layers.Dense(self.n_hidden, activation=self.non_linearity)(self.inputs)
        self.h_mean = tf.keras.layers.Dense(self.n_topic)(self.enc_vec)
        self.h_logsigm = tf.keras.layers.Dense(self.n_topic)(self.enc_vec)
        self.encoder = tf.keras.Model(self.inputs, self.h_mean)

    def build_prior_net(self):
        self.p_mean = 0.
        self.p_logsigm = 0.

    def build_generative_net(self):
        self.latent_inputs = tf.keras.Input(shape=(self.n_topic,))
        logits = tf.keras.layers.Dense(self.vocab_size, activation=tf.nn.log_softmax)(self.latent_inputs)
        self.decoder = tf.keras.Model(self.latent_inputs, logits)

        eps = tf.random.normal(tf.shape(self.h_mean), 0., 1.)
        self.doc_vec = self.h_mean + eps * tf.exp(self.h_logsigm)
        self.logits = self.decoder(self.doc_vec)
        self.topic_word_dist = self.decoder.weights[0]

class LDAWordGenerator(tf.keras.layers.Layer):
    def __init__(self, vocab_size, drop_prob, beta=None, name="LDAWordGenerator", **kwargs):
        super(LDAWordGenerator, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.drop_prob = drop_prob
        self.beta = beta
        if beta is None:
            self.beta = 1 / self.vocab_size

    def build(self, input_shape):
        stddev = (1 / self.beta * (1 - 1 / self.vocab_size)) ** 0.5 # simulate a dirichlet prior with lognormal
        self.w = self.add_weight(shape=(input_shape[-1], self.vocab_size),
            initializer=tf.keras.initializers.RandomNormal(0, stddev), trainable=True)
        # self.w = self.add_weight(shape=(input_shape[-1], self.vocab_size),
        #     initializer='random_normal', trainable=True)

    def call(self, inputs, training=None):
        if training:
            inputs = tf.nn.dropout(inputs, rate=self.drop_prob) #[batch_size, n_topics]
        inputs = inputs - tf.reduce_logsumexp(inputs, axis=1, keepdims=True) #[batch_size, n_topics]
        w = self.w - tf.reduce_logsumexp(self.w, axis=1, keepdims=True) #[n_topics, vocab_size]
        z = tf.reduce_logsumexp(tf.expand_dims(inputs, 2) + tf.expand_dims(w, 0), axis=1) #[batch_size, vocab_size]
        return z

    def get_config(self):
        config = super(LDAWordGenerator, self).get_config()
        config.update({"vocab_size" : self.vocab_size, "drop_prob": drop_prob})
        return config

class ProdLDAWordGenerator(tf.keras.layers.Layer):
    def __init__(self, vocab_size, drop_prob, wlogits=True, name="ProdLDAWordGenerator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.drop_prob = drop_prob
        self.wlogits = wlogits

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.vocab_size),
            initializer='random_normal', trainable=True)
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        if training:
            inputs = tf.nn.dropout(inputs, rate=self.drop_prob)
        if self.wlogits:
            inputs = tf.nn.softmax(inputs)
        x = tf.matmul(inputs, self.w)
        x = self.batch_norm(x)
        return tf.nn.log_softmax(x)

    def get_config(self):
        config = super(LDAWordGenerator, self).get_config()
        config.update({"vocab_size" : self.vocab_size,
            "drop_prob": self.drop_prob, "wlogits" : self.wlogits})
        return config

class LDA(BaseNVDM):
    """LDA model the same as https://arxiv.org/abs/1703.01488
    """
    def __init__(self, vocab_size, n_topic, n_hidden, non_linearity, drop_prob, alpha=None, **kwargs):
        self._init(vocab_size, n_topic, **kwargs)
        self.n_hidden = n_hidden
        self.non_linearity = non_linearity
        if isinstance(non_linearity, str):
            print("non_linearity", non_linearity)
            self.non_linearity = eval(non_linearity)
        self.drop_prob = drop_prob
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = 1. / self.n_topic
        self.build()

    def build_inference_net(self):
        enc_vec = self.inputs
        for n_hidden_i in self.n_hidden:
            enc_vec = tf.keras.layers.Dense(n_hidden_i, activation=self.non_linearity)(enc_vec)
        self.enc_vec = tf.keras.layers.Dropout(self.drop_prob)(enc_vec)
        h_mean = tf.keras.layers.Dense(self.n_topic)(self.enc_vec)
        self.h_mean = tf.keras.layers.BatchNormalization()(h_mean)
        h_logsigm = tf.keras.layers.Dense(self.n_topic)(self.enc_vec)
        self.h_logsigm = tf.keras.layers.BatchNormalization()(h_logsigm)
        self.encoder = tf.keras.Model(self.inputs, self.h_mean)

    def build_prior_net(self):
        self.p_mean = 0
        sigma2 = 1. / self.alpha * (1 - 2. / self.n_topic) + 1. / self.n_topic / self.alpha #self.n_topic - 1
        self.p_logsigm = math.log(sigma2) / 2

    def build_generative_net(self):
        self.latent_inputs = tf.keras.Input(shape=(self.n_topic,))
        logits = LDAWordGenerator(self.vocab_size, self.drop_prob)(self.latent_inputs)
        self.decoder = tf.keras.Model(self.latent_inputs, logits)

        eps = tf.random.normal(tf.shape(self.h_mean), 0., 1.)
        self.doc_vec = self.h_mean + eps * tf.exp(self.h_logsigm)
        self.logits = self.decoder(self.doc_vec)
        self.topic_word_dist = self.decoder.weights[0]

class ProdLDA(LDA):
    """ProdLDA model the same as https://arxiv.org/abs/1703.01488
    """
    def build_generative_net(self):
        self.latent_inputs = tf.keras.Input(shape=(self.n_topic,))
        logits = ProdLDAWordGenerator(self.vocab_size, self.drop_prob)(self.latent_inputs)
        self.decoder = tf.keras.Model(self.latent_inputs, logits)

        eps = tf.random.normal(tf.shape(self.h_mean), 0., 1.)
        self.doc_vec = self.h_mean + eps * tf.exp(self.h_logsigm)
        self.logits = self.decoder(self.doc_vec)
        self.topic_word_dist = self.decoder.weights[0]


class ProdLDAwMeta(ProdLDA):
    """
    Similar to ProdLDA, in addition to generating text, the topics can also be used
    to generate meta data. Here treat discrete meta data similarly as words and we
    learn distributions of meta data given topic.
    """
    def __init__(self, vocab_size, n_topic, n_hidden, non_linearity,
        metanet_n_hidden, metanet_non_linearity, x_num_classes,
        drop_prob, alpha=None, **kwargs):
        self.metanet_n_hidden = metanet_n_hidden
        self.metanet_non_linearity = metanet_non_linearity
        if isinstance(metanet_non_linearity, str):
            print("metanet_non_linearity", metanet_non_linearity)
            self.metanet_non_linearity = eval(metanet_non_linearity)
        self.x_num_classes = x_num_classes
        super().__init__(vocab_size, n_topic, n_hidden, non_linearity,
            drop_prob, alpha, supervised=True, **kwargs)

    def build_inference_net(self):
        inputs_x = self.inputs_x # assuming inputs_x in (batch_size, x_num_classes) shape
        for n_hidden_i in self.metanet_n_hidden:
            inputs_x = tf.keras.layers.Dense(n_hidden_i, activation=self.metanet_non_linearity)(inputs_x)
        inputs_x = tf.keras.layers.Dense(self.n_topic, activation=self.metanet_non_linearity)(inputs_x)

        enc_vec = self.inputs
        for n_hidden_i in self.n_hidden:
            enc_vec = tf.keras.layers.Dense(n_hidden_i, activation=self.non_linearity)(enc_vec)
        enc_vec = tf.keras.layers.Dense(self.n_topic, activation=self.non_linearity)(enc_vec)
        enc_vec = tf.keras.layers.Concatenate()([inputs_x, enc_vec])
        enc_vec = tf.keras.layers.Dropout(self.drop_prob)(enc_vec)

        h_mean = tf.keras.layers.Dense(self.n_topic)(enc_vec)
        self.h_mean = tf.keras.layers.BatchNormalization()(h_mean)
        h_logsigm = tf.keras.layers.Dense(self.n_topic)(enc_vec)
        self.h_logsigm = tf.keras.layers.BatchNormalization()(h_logsigm)
        self.encoder = tf.keras.Model([self.inputs, self.inputs_x], self.h_mean)
        return

    def build_generative_net(self):
        super().build_generative_net()
        # self.encoder = tf.keras.Model([self.inputs, self.inputs_x], self.doc_vec)
        meta_logits = LDAWordGenerator(self.x_num_classes, self.drop_prob)(self.latent_inputs)
        self.meta_decoder = tf.keras.Model(self.latent_inputs, meta_logits)
        self.meta_logits = self.meta_decoder(self.doc_vec)

    def get_prior_topic_dist(self):
        return self.meta_decoder.weights[0].numpy()

    def get_recon_loss(self):
        super().get_recon_loss()
        self.recon_loss -= tf.reduce_sum(self.meta_logits * self.inputs_x, 1)

class AuthorTopicModel(ProdLDA):
    """
    neural variational implementation of topic model as described in
    https://arxiv.org/abs/1207.4169
    """
    def __init__(self, vocab_size, n_topic, n_hidden, non_linearity,
        priornet_n_hidden, priornet_non_linearity,
        x_num_classes, drop_prob, author_alpha=None, alpha=None, **kwargs):
        self.x_num_classes = x_num_classes
        self.priornet_n_hidden = priornet_n_hidden
        self.priornet_non_linearity = priornet_non_linearity
        self.author_alpha = author_alpha
        if isinstance(priornet_non_linearity, str):
            print("priornet_non_linearity", priornet_non_linearity)
            self.priornet_non_linearity = eval(priornet_non_linearity)
        super().__init__(vocab_size, n_topic, n_hidden, non_linearity,
            drop_prob, alpha, supervised=True, **kwargs)

    def get_prior_topic_dist(self):
        return self.prior_topic_dist.numpy() # [x_num_classes, n_topic]

    def get_kld(self):
        self.kld = kld((self.h_mean, self.h_logsigm), (self.p_mean, self.p_logsigm),
            self.variational_dist, self.prior_dist, mask=tf.not_equal(self.h_mean, 0))

    def build_prior_net(self):
        inputs_x = self.inputs_x # assuming inputs_x in (batch_size, x_num_classes) shape
        n_authors = tf.reduce_sum(inputs_x, axis=1, keepdims=True) + 1
        self.p_mean = 0
        sigma2 = 1. / self.author_alpha * (1 - 2. / n_authors) + 1. / n_authors / self.author_alpha #self.n_topic - 1
        self.p_logsigm = tf.math.log(sigma2) / 2

    def build_inference_net(self):
        enc_vec = self.inputs
        for n_hidden_i in self.n_hidden:
            enc_vec = tf.keras.layers.Dense(n_hidden_i, activation=self.non_linearity)(enc_vec)
        enc_vec = tf.keras.layers.Dense(self.x_num_classes, activation=self.non_linearity)(enc_vec)
        enc_vec = tf.keras.layers.Dropout(self.drop_prob)(enc_vec)

        h_mean = tf.keras.layers.Dense(self.x_num_classes)(enc_vec)
        self.h_mean = tf.keras.layers.BatchNormalization()(h_mean) * self.inputs_x#self.inputs_x_onehot
        h_logsigm = tf.keras.layers.Dense(self.x_num_classes)(enc_vec)
        self.h_logsigm = tf.keras.layers.BatchNormalization()(h_logsigm) * self.inputs_x#self.inputs_x_onehot
        return

    def build_generative_net(self):
        latent_inputs_logits = tf.keras.Input(shape=(self.x_num_classes,))
        topic_logits = LDAWordGenerator(self.n_topic, self.drop_prob, beta=self.alpha)(latent_inputs_logits)#, beta=self.alpha
        logits = ProdLDAWordGenerator(self.vocab_size, self.drop_prob)(topic_logits)
        self.decoder = tf.keras.Model(latent_inputs_logits, logits)

        eps = tf.random.normal(tf.shape(self.h_mean), 0., 1.)
        self.doc_vec = self.h_mean + eps * tf.exp(self.h_logsigm)
        self.encoder = tf.keras.Model([self.inputs, self.inputs_x], self.doc_vec)
        self.logits = self.decoder(self.doc_vec)
        self.topic_word_dist = self.decoder.weights[1]
        self.prior_topic_dist = self.decoder.weights[0]

import numpy as np
import tensorflow as tf
import argparse
import os
import shutil
import time
import sys
import logging
import json
from functools import reduce
from operator import mul
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from tensorflow.python.util import nest
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from dataset import load_vocab

import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(12345)

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None, strict=None):
        if strict is None:
            strict = []
        if exact is None:
            exact = []
        if values is None:
            values = []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)
            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])
            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * ' ')
            sys.stdout.write(info)
            sys.stdout.flush()
            if current >= self.target:
                sys.stdout.write("\n")
        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        if values is None:
            values = []
        self.update(self.seen_so_far+n, values)
        
def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except IOError:
        raise "ERROR: Unable to locate file {}".format(filename)


def load_embeddings(filename):
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise 'ERROR: Unable to locate file {}.'.format(filename)


def batch_iter(dataset, batch_size):
    batch_x, batch_y = [], []
    for record in dataset:
        if len(batch_x) == batch_size:
            yield batch_x, batch_y
            batch_x, batch_y = [], []
        x = [tuple(value) for value in record["sentence"]]
        x = zip(*x)
        y = record["label"]
        batch_x += [x]
        batch_y += [y]
    if len(batch_x) != 0:
        yield batch_x, batch_y


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        if len(seq) < max_length:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        else:
            seq_ = seq[:max_length]
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


def pad_sequences(sequences, max_length, pad_tok, max_length_2=None, nlevels=1):
    sequence_padded, sequence_length = [], []
    if nlevels == 1:
        if max_length is None:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        if max_length_2 is None:
            max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_2)
            sequence_padded += [sp]
            sequence_length += [sl]
        if max_length is None:
            max_length_sentence = max(map(lambda x: len(x), sequences))
        else:
            max_length_sentence = max_length
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_2, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length

def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=None, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[filter_size], dtype=tf.float32)
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob is not None:
            in_ = dropout(in_, keep_prob, is_train)
        # [batch, max_len_sent, max_len_word / filter_stride, char output size]
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias
        out = tf.reduce_max(tf.nn.relu(xxc), axis=2)  # max-pooling, [-1, max_len_sent, char output size]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=None, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for i, (filter_size, height) in enumerate(zip(filter_sizes, heights)):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob,
                         scope="conv1d_{}".format(i))
            outs.append(out)
        concat_out = tf.concat(axis=2, values=outs)
        return concat_out

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if keep_prob is not None and is_train is not None:
            out = tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed), lambda: x)
            return out
        return x
    
class BiRNN:
    def __init__(self, num_units, scope='bi_rnn'):
        self.num_units = num_units
        self.cell_fw = LSTMCell(self.num_units)
        self.cell_bw = LSTMCell(self.num_units)
        self.scope = scope

    def __call__(self, inputs, seq_len, return_last_state=False):
        with tf.variable_scope(self.scope):
            if return_last_state:
                _, ((_, output_fw), (_, output_bw)) = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                                sequence_length=seq_len,
                                                                                dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
            else:
                (output_fw, output_bw), _ = bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs,
                                                                      sequence_length=seq_len, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
        return output

def dense(inputs, hidden_dim, use_bias=True, scope='dense'):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_dim]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        w = tf.get_variable("weight", shape=[dim, hidden_dim], dtype=tf.float32)
        output = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("bias", shape=[hidden_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)
        output = tf.reshape(output, out_shape)
        return output
    
class DenseConnectBiLSTM(object):
    def __init__(self, config, resume_training=True):


        # set configurations
        self.cfg, self.model_name, self.resume_training, self.start_epoch = config, config.model_name, resume_training, 1
        self.logger = get_logger(os.path.join(self.cfg.ckpt_path, 'log.txt'))
        # build model
        self._add_placeholder()
        self._add_embedding_lookup()
        self._build_model()
        self._add_loss_op()
        self._add_accuracy_op()
        self._add_train_op()
        print('params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        # initialize model
        self.sess, self.saver = None, None
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())
        if self.resume_training:
            checkpoint = tf.train.get_checkpoint_state(self.cfg.ckpt_path)
            return    
            print('Resume training from %s...' % self.cfg.ckpt_path)
            ckpt_path = checkpoint.model_checkpoint_path
            self.start_epoch = int(ckpt_path.split('-')[-1]) + 1
            print('Start Epoch: ', self.start_epoch)
            self.saver.restore(self.sess, ckpt_path)

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        if not os.path.exists(self.cfg.ckpt_path):
            os.makedirs(self.cfg.dir_model)
        self.saver.save(self.sess, self.cfg.ckpt_path + self.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_placeholder(self):
        # shape = (batch_size, max_sentence_length)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        # shape = (batch_size)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        # shape = (batch_size, max_sentence_length, max_word_length)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_ids')
        # shape = (batch_size, max_sentence_length)
        self.word_len = tf.placeholder(tf.int32, shape=[None, None], name='word_len')
        # shape = (batch_size, label_size)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        # hyper-parameters
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, words, labels=None, lr=None, is_train=None):
        word_ids, char_ids = zip(*words)
        word_ids, seq_len = pad_sequences(word_ids, max_length=None, pad_tok=0, nlevels=1)
        feed_dict = {self.word_ids: word_ids, self.seq_len: seq_len}
        if self.cfg.use_char_emb:
            char_ids, word_len = pad_sequences(char_ids, max_length=None, pad_tok=0, max_length_2=None, nlevels=2)
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_len] = word_len
        if labels is not None:
            feed_dict[self.labels] = labels
        if lr is not None:
            feed_dict[self.lr] = lr
        if is_train is not None:
            feed_dict[self.is_train] = is_train
        return feed_dict

    def _add_embedding_lookup(self):
        with tf.variable_scope('word_embeddings'):
            if self.cfg.use_word_emb:
                _word_emb = tf.Variable(self.cfg.word_emb, name='_word_emb', trainable=self.cfg.finetune_emb,
                                        dtype=tf.float32)
            else:
                _word_emb = tf.get_variable(name='_word_emb', shape=[self.cfg.vocab_size, self.cfg.word_dim],
                                            trainable=True, dtype=tf.float32)
            word_emb = tf.nn.embedding_lookup(_word_emb, self.word_ids, name='word_emb')

        if self.cfg.use_char_emb:  # use cnn to generate chars representation
            with tf.variable_scope('char_embeddings'):
                _char_emb = tf.get_variable(name='_char_emb', dtype=tf.float32, trainable=True,
                                            shape=[self.cfg.char_vocab_size, self.cfg.char_dim])
                char_emb = tf.nn.embedding_lookup(_char_emb, self.char_ids, name='char_emb')
                char_emb_shape = tf.shape(char_emb)
                char_rep = multi_conv1d(char_emb, self.cfg.filter_sizes, self.cfg.heights, "VALID",  self.is_train,
                                        self.cfg.keep_prob, scope="char_cnn")
                char_rep = tf.reshape(char_rep, [char_emb_shape[0], char_emb_shape[1], self.cfg.char_rep_dim])
                word_emb = tf.concat([word_emb, char_rep], axis=-1)  # concat word emb and corresponding char rep

        self.word_emb = dropout(word_emb, keep_prob=self.cfg.keep_prob, is_train=self.is_train)
        print('word embedding shape: {}'.format(self.word_emb.get_shape().as_list()))

    def _build_model(self):
        with tf.variable_scope('dense_connect_bi_lstm'):
            # create dense connected bi-lstm layers
            dense_bi_lstm = []
            for idx in range(self.cfg.num_layers):
                if idx < self.cfg.num_layers - 1:
                    dense_bi_lstm.append(BiRNN(num_units=self.cfg.num_units, scope='bi_lstm_layer_{}'.format(idx)))
                else:
                    dense_bi_lstm.append(BiRNN(num_units=self.cfg.num_units_last, scope='bi_lstm_layer_{}'.format(idx)))
            # processing data
            cur_inputs = self.word_emb
            for idx in range(self.cfg.num_layers):
                cur_rnn_outputs = dense_bi_lstm[idx](cur_inputs, seq_len=self.seq_len)
                if self.cfg.model_name == 'DC_Bi_LSTM':  # Adjust input of all layers for DC_Bi_LSTM 
                    if idx < self.cfg.num_layers - 1:
                        cur_inputs = tf.concat([cur_inputs, cur_rnn_outputs], axis=-1)
                    else:
                        cur_inputs = cur_rnn_outputs
                else:                                    # Adjust input of all layers for other models
                    cur_inputs = cur_rnn_outputs     
            dense_bi_lstm_outputs = cur_inputs
            print('dense bi-lstm outputs shape: {}'.format(dense_bi_lstm_outputs.get_shape().as_list()))

        with tf.variable_scope('average_pooling'):
            # according to the paper (https://arxiv.org/pdf/1802.00889.pdf) description in P4, simply compute average ?
            avg_pool_outputs = tf.reduce_mean(dense_bi_lstm_outputs, axis=1)
            avg_pool_outputs = dropout(avg_pool_outputs, keep_prob=self.cfg.keep_prob, is_train=self.is_train)
            print('average pooling outputs shape: {}'.format(avg_pool_outputs.get_shape().as_list()))

        with tf.variable_scope('project', regularizer=tf.contrib.layers.l2_regularizer(self.cfg.l2_reg)):
            self.logits = dense(avg_pool_outputs, self.cfg.label_size, use_bias=True, scope='dense')
            print('logits shape: {}'.format(self.logits.get_shape().as_list()))

    def _add_loss_op(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(loss) + 0.5 * l2_loss

    def _add_accuracy_op(self):
        self.predicts = tf.argmax(self.logits, axis=-1)
        self.actuals = tf.argmax(self.labels, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, self.actuals), dtype=tf.float32))

    def _add_train_op(self):
        with tf.variable_scope('train_step'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            if self.cfg.grad_clip is not None:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, trainset, devset, testset, batch_size=64, epochs=50, shuffle=True):
        self.logger.info('Start training...')
        init_lr = self.cfg.lr  # initial learning rate, used for decay learning rate
        best_score = 0.0  # record the best score
        best_score_epoch = 1  # record the epoch of the best score obtained
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            progbar = Progbar(target=(len(trainset) + batch_size - 1) // batch_size)  # number of batches
            if shuffle:
                np.random.shuffle(trainset)  # shuffle training dataset each epoch
            # training each epoch
            for i, (words, labels) in enumerate(batch_iter(trainset, batch_size)):
                feed_dict = self._get_feed_dict(words, labels, lr=self.cfg.lr, is_train=True)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                progbar.update(i + 1, [("train loss", train_loss)])
            if devset is not None:
                self.evaluate(devset, batch_size)
            cur_score = self.evaluate(testset, batch_size, is_devset=False)
            # learning rate decay
            if self.cfg.decay_lr:
                self.cfg.lr = init_lr / (1 + self.cfg.lr_decay * epoch)
            # performs model saving and evaluating on test dataset
            if cur_score > best_score:
                self.save_session(epoch)
                best_score = cur_score
                best_score_epoch = epoch
                self.logger.info(' -- new BEST score on TEST dataset: {:05.3f}'.format(best_score))
            else:
                self.logger.info('BEST score: ''{:05.3f} at epoch {}'.format(best_score, best_score_epoch))
        self.logger.info('Training process done...')

    def evaluate(self, dataset, batch_size, is_devset=True):
        accuracies = []
        for words, labels in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(words, labels, lr=None, is_train=False)
            accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
            accuracies.append(accuracy)
        acc = np.mean(accuracies) * 100
        self.logger.info("Testing model over {} dataset: accuracy - {:05.3f}".format('DEVELOPMENT' if is_devset else
                                                                                     'TEST', acc))
        return acc

class Config(object):
    def __init__(self, task, model_name):
        self.ckpt_path = './ckpt/{}/'.format(task)
        if os.path.exists(self.ckpt_path):
            shutil.rmtree(self.ckpt_path)
        os.makedirs(self.ckpt_path)
        source_dir = os.path.join('.', 'dataset', 'data', task)
        self.word_vocab, _ = load_vocab(os.path.join(source_dir, 'words.vocab'))
        self.char_vocab, _ = load_vocab(os.path.join(source_dir, 'chars.vocab'))
        self.vocab_size = len(self.word_vocab)
        self.char_vocab_size = len(self.char_vocab)
        self.label_size = load_json(os.path.join(source_dir, 'label.json'))["label_size"]
        self.word_emb = load_embeddings(os.path.join(source_dir, 'glove.filtered.npz'))
        self.model_name = model_name
        #print(model_name)
        if model_name == 'DC_Bi_LSTM':
            self.num_layers = 15       # 15
            self.num_units = 13        # 13
            self.num_units_last = 100      
        elif model_name == 'DS_Bi_LSTM':
            self.num_layers = 5       # 15
            self.num_units = 85        # 13
            self.num_units_last = 200
        elif model_name == 'Bi_LSTM':
            self.num_layers = 1       # 15
            self.num_units = 40        # 13
            self.num_units_last = 300
        else:
            print('Invalid model name')
            exit(0)

    # log and model file paths
    max_to_keep = 40  # max model to keep while training

    # word embeddings
    use_word_emb = True
    finetune_emb = False
    word_dim = 300

    # char embeddings
    use_char_emb = True
    char_dim = 50
    char_rep_dim = 50
    # Convolutional neural networks filter size and height for char representation
    filter_sizes = [25, 25]  # sum of filter sizes should equal to char_out_size
    heights = [5, 5]

    # hyperparameters
    l2_reg = 0.001
    grad_clip = 5.0
    decay_lr = True
    lr = 0.01
    lr_decay = 0.05
    keep_prob = 0.5

def main():
    data_folder = os.path.join('.', 'dataset', 'data')
    # set tasks
    source_dir = os.path.join(data_folder, task)
    # create config
    config = Config(task, model_name)
    print(config.num_layers)
    # load datasets
    trainset = load_json(os.path.join(source_dir, 'train.json'))
    devset = load_json(os.path.join(source_dir, 'dev.json'))
    testset = load_json(os.path.join(source_dir, 'test.json'))
    # build model
    model = DenseConnectBiLSTM(config, resume_training=resume_training)
    # training
    batch_size = 200
    epochs = 30
    if has_devset:
        model.train(trainset, devset, testset, batch_size=batch_size, epochs=epochs, shuffle=True)
    else:
        trainset = trainset + devset
        model.train(trainset, None, testset, batch_size=batch_size, epochs=epochs, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='set train task (cr|mpqa|mr|sst1|sst2|subj|trec).')
    parser.add_argument('--model', type=str, required=True, help='set model (DC_Bi_LSTM|DS_Bi_LSTM|Bi_LSTM).')
    parser.add_argument('--has_devset', type=str, required=True, help='indicates if the task has development dataset.')
    
    args, _ = parser.parse_known_args()
    task = args.task
    model_name = args.model
    resume_training = True
    has_devset = True if args.has_devset == 'True' else False
    main()

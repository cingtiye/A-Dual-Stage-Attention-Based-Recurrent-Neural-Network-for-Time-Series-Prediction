# -*- coding: utf-8

import numpy as np
import config
import time
import tensorflow as tf
tf.set_random_seed(2019)
import warnings
from model.lookahead_optimizer import LookaheadOptimizer
from model.radam_optimizer import RAdamOptimizer
warnings.filterwarnings('ignore')

class Attention(object):
    ## 类初始化
    def __init__(self,
                 INPUT_LENGTH = 81,
                 TIME_STEP    = 10,
                 ENCODE_CELL  = 64,
                 DECODE_CELL  = 64,
                 LEARN_RATE   = 0.001,
                 BATCH_SIZE   = 128, ):

        # Definition Params:
        self.INPUT_LENGTH = INPUT_LENGTH  # Sensor Number * Feature Number
        self.TIME_STEP = TIME_STEP        # Window Number
        self.ENCODE_CELL = ENCODE_CELL    # LSTM Encode Cell Units (int)
        self.DECODE_CELL = DECODE_CELL    # LSTM Decode Cell Units (int)
        self.LEARN_RATE = LEARN_RATE      # Original learning rate value
        self.BATCH_SIZE = BATCH_SIZE      # batch size

        # Define Net Input:
        self.X = tf.placeholder(shape=[None, self.TIME_STEP, self.INPUT_LENGTH],
                                dtype=tf.float32, name='input_x')
        self.encode_hidden = tf.Variable(tf.constant(0., shape=[self.BATCH_SIZE, self.ENCODE_CELL]),
                                         dtype=tf.float32, trainable=False, name='encode_hidden_state')
        self.encode_cell = tf.Variable(tf.constant(0., shape=[self.BATCH_SIZE, self.ENCODE_CELL]),
                                       dtype=tf.float32, trainable=False, name='encode_cell_state')
        self.decode_hidden = tf.Variable(tf.constant(0., shape=[self.BATCH_SIZE, self.DECODE_CELL]),
                                         dtype=tf.float32, trainable=False, name='decode_hidden_state')
        self.decode_cell = tf.Variable(tf.constant(0., shape=[self.BATCH_SIZE, self.DECODE_CELL]),
                                       dtype=tf.float32, trainable=False, name='decode_cell_state')
        self.history_y = tf.placeholder(shape=[None, self.TIME_STEP-1], dtype=tf.float32, name='history_label')
        self.y = tf.placeholder(shape=[None, ], dtype=tf.float32, name='true_y')
        # self.learning_rate = tf.placeholder(shape=None, dtype=tf.float32, name='original_learning_rate')

        # Define Weights and Bias:
        self._weights = self._initialize_weights()

        # Construct Network:
        with tf.variable_scope('Encode', reuse=tf.AUTO_REUSE):
            self.Encode_hidden_state = self._Encode(self.X)
            tf.summary.histogram('Encode_hidden_state', self.Encode_hidden_state)
        with tf.variable_scope('Decode', reuse=tf.AUTO_REUSE):
            self.pre_y = self._Decode(self.Encode_hidden_state, self.history_y)
            tf.summary.histogram('pre_y', self.pre_y)
        with tf.variable_scope('cost-function', reuse=tf.AUTO_REUSE):
            self._create_cost_function(self.pre_y)

    ## 初始化论文中所有的参数(权重和偏置)
    def _initialize_weights(self):
        _dic_weights = dict()

        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL+self.INPUT_LENGTH,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Encode_LSTMCell_ForgetGate',
                                        _is_bias    = True, )
        _dic_weights['Encode_LSTMCell_ForgetGate_weights'] = _weights[0]
        _dic_weights['Encode_LSTMCell_ForgetGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL + self.INPUT_LENGTH,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Encode_LSTMCell_InputGate',
                                        _is_bias    = True, )
        _dic_weights['Encode_LSTMCell_InputGate_weights'] = _weights[0]
        _dic_weights['Encode_LSTMCell_InputGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL + self.INPUT_LENGTH,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Encode_LSTMCell_OutputGate',
                                        _is_bias    = True, )
        _dic_weights['Encode_LSTMCell_OutputGate_weights'] = _weights[0]
        _dic_weights['Encode_LSTMCell_OutputGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL + self.INPUT_LENGTH,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Encode_LSTMCell_CellGate',
                                        _is_bias    = True, )
        _dic_weights['Encode_LSTMCell_CellGate_weights'] = _weights[0]
        _dic_weights['Encode_LSTMCell_CellGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL*2,
                                        _output_dim = self.TIME_STEP,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Input_attention_layer_We',
                                        _is_bias    = False, )
        _dic_weights['Input_attention_layer_We'] = _weights[0]
        _weights = self._define_weights(_input_dim  = self.TIME_STEP,
                                        _output_dim = self.TIME_STEP,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Input_attention_layer_Ue',
                                        _is_bias    = False, )
        _dic_weights['Input_attention_layer_Ue'] = _weights[0]
        _weights = self._define_weights(_input_dim  = self.TIME_STEP,
                                        _output_dim = 1,
                                        _activation = None,
                                        _name       = 'Input_attention_layer_Ve',
                                        _is_bias    = False, )
        _dic_weights['Input_attention_layer_Ve'] = _weights[0]

        _weights = self._define_weights(_input_dim  = self.DECODE_CELL + 1,
                                        _output_dim = self.DECODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Decode_LSTMCell_ForgetGate',
                                        _is_bias    = True, )
        _dic_weights['Decode_LSTMCell_ForgetGate_weights'] = _weights[0]
        _dic_weights['Decode_LSTMCell_ForgetGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.DECODE_CELL + 1,
                                        _output_dim = self.DECODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Decode_LSTMCell_InputGate',
                                        _is_bias    = True, )
        _dic_weights['Decode_LSTMCell_InputGate_weights'] = _weights[0]
        _dic_weights['Decode_LSTMCell_InputGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.DECODE_CELL + 1,
                                        _output_dim = self.DECODE_CELL,
                                        _activation = tf.nn.sigmoid,
                                        _name       = 'Decode_LSTMCell_OutputGate',
                                        _is_bias    = True, )
        _dic_weights['Decode_LSTMCell_OutputGate_weights'] = _weights[0]
        _dic_weights['Decode_LSTMCell_OutputGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.DECODE_CELL + 1,
                                        _output_dim = self.DECODE_CELL,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Decode_LSTMCell_CellGate',
                                        _is_bias    = True, )
        _dic_weights['Decode_LSTMCell_CellGate_weights'] = _weights[0]
        _dic_weights['Decode_LSTMCell_CellGate_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.DECODE_CELL * 2,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Temporal_attention_layer_Wd',
                                        _is_bias    = False, )
        _dic_weights['Temporal_attention_layer_Wd'] = _weights[0]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL,
                                        _output_dim = self.ENCODE_CELL,
                                        _activation = tf.nn.tanh,
                                        _name       = 'Temporal_attention_layer_Ud',
                                        _is_bias    = False, )
        _dic_weights['Temporal_attention_layer_Ud'] = _weights[0]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL,
                                        _output_dim = 1,
                                        _activation = None,
                                        _name       = 'Temporal_attention_layer_Vd',
                                        _is_bias    = False, )
        _dic_weights['Temporal_attention_layer_Vd'] = _weights[0]

        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL+1,
                                        _output_dim = 1,
                                        _activation = None,
                                        _name       = 'Decode_layer_yt',
                                        _is_bias    = True, )
        _dic_weights['Decode_layer_yt_weights'] = _weights[0]
        _dic_weights['Decode_layer_yt_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.ENCODE_CELL + self.DECODE_CELL,
                                        _output_dim = self.DECODE_CELL,
                                        _activation = None,
                                        _name       = 'Decode_layer_output_1',
                                        _is_bias    = True, )
        _dic_weights['Decode_layer_output_1_weights'] = _weights[0]
        _dic_weights['Decode_layer_output_1_bias'] = _weights[1]
        _weights = self._define_weights(_input_dim  = self.DECODE_CELL,
                                        _output_dim = 1,
                                        _activation = None,
                                        _name       = 'Decode_layer_output_2',
                                        _is_bias    = True, )
        _dic_weights['Decode_layer_output_2_weights'] = _weights[0]
        _dic_weights['Decode_layer_output_2_bias'] = _weights[1]

        for keys, values in _dic_weights.items():
            self.variable_summaries(values, keys)
            if 'weights' in keys.split('_'):
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, values)
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer)
        tf.summary.scalar('reg_cost', self.reg_term)
        return _dic_weights

    ## 单个LSTM单元的实现
    def _LSTMCell(self,
                  h_t_1 = None,    # LSTM hidden state shape: [-1, cell_dim]
                  s_t_1 = None,    # LSTM cell state shape: [-1, cell_dim]
                  x_t   = None,    # LSTM input shape: [-1, input_dim]
                  name  = None, ): # LSTM name: Encode_LSTMCell or Decode_LSTMCell
        _input = tf.concat([h_t_1, x_t], axis=1)  # [-1, cell_dim+input_dim]
        forget_t = self._Dense(_input      = _input,
                               _weights    = self._weights[name + '_ForgetGate_weights'],
                               _bias       = self._weights[name + '_ForgetGate_bias'],
                               _activation = tf.nn.sigmoid,
                               _dtype      = tf.float32,
                               _is_bias    = True, )
        input_t = self._Dense(_input       = _input,
                              _weights     = self._weights[name + '_InputGate_weights'],
                              _bias        = self._weights[name + '_InputGate_bias'],
                              _activation  = tf.nn.sigmoid,
                              _dtype       = tf.float32,
                              _is_bias     = True, )
        output_t = self._Dense(_input      = _input,
                               _weights    = self._weights[name + '_OutputGate_weights'],
                               _bias       = self._weights[name + '_OutputGate_bias'],
                               _activation = tf.nn.sigmoid,
                               _dtype      = tf.float32,
                               _is_bias    = True, )
        s_t = tf.add(tf.multiply(forget_t, s_t_1),
                     tf.multiply(input_t, self._Dense(
                         _input      = _input,
                         _weights    = self._weights[name + '_CellGate_weights'],
                         _bias       = self._weights[name + '_CellGate_bias'],
                         _activation = tf.nn.tanh,
                         _dtype      = tf.float32,
                         _is_bias    = True, )), name=name + '_LSTM_update_cell_state')
        h_t = tf.multiply(output_t, tf.nn.tanh(s_t), name=name + '_LSTM_update_hidden_state')
        return (s_t, h_t)

    ## 编码器中加入Attention
    def _Encode(self, encode_input=None):  # encode_input: [-1, time_step, input_dim]
        x_k = tf.transpose(encode_input, perm=[0, 2, 1], name='Series_of_length_TIME_STEP')#[-1,input_dim,time_step]
        encode_time_step_hidden = []
        for t in range(encode_input.get_shape()[1]):  # [t < time_step]
            e_t = self._attention_layer(_h_t_1 = self.encode_hidden,
                                        _s_t_1 = self.encode_cell,
                                        _x_k   = x_k,
                                        _We    = self._weights['Input_attention_layer_We'],
                                        _Ue    = self._weights['Input_attention_layer_Ue'],
                                        _Ve    = self._weights['Input_attention_layer_Ve'], )
            a_t = tf.nn.softmax(e_t)  # [-1, input_dim]
            tmp = tf.reshape(encode_input[:, t, :], shape=[-1, encode_input.get_shape().as_list()[-1]])
            x_t = tf.multiply(a_t, tmp)
            (self.encode_cell, self.encode_hidden) = self._LSTMCell(h_t_1 = self.encode_hidden,
                                                                    s_t_1 = self.encode_cell,
                                                                    x_t   = x_t,
                                                                    name  = 'Encode_LSTMCell')
            encode_time_step_hidden.append(self.encode_hidden)
        return tf.reshape(tf.stack(encode_time_step_hidden), [-1, self.TIME_STEP, self.DECODE_CELL])

    ## 解码器中加入Attention
    def _Decode(self, decode_input=None, y_t=None):
        for t in range(decode_input.get_shape()[1]-1):
            l_t = self._attention_layer(_h_t_1 = self.decode_hidden,
                                        _s_t_1 = self.decode_cell,
                                        _x_k   = decode_input,
                                        _We    = self._weights['Temporal_attention_layer_Wd'],
                                        _Ue    = self._weights['Temporal_attention_layer_Ud'],
                                        _Ve    = self._weights['Temporal_attention_layer_Vd'], )
            b_t = tf.reshape(tf.nn.softmax(l_t), shape=[-1, decode_input.get_shape().as_list()[1], 1])  # [-1, time_step, 1]
            c_t = tf.reduce_sum(tf.multiply(b_t, decode_input), axis=1)  # [-1, time_step, 1]*[-1, time_step, cell_dim]
                                                                         # ---> [-1, time_step, cell_dim]-->[-1, cell_dim]
            y_t_ = self._Dense(_input       = tf.concat([c_t, tf.reshape(y_t[:, t], [-1, 1])], axis=1),
                               _weights     = self._weights['Decode_layer_yt_weights'],
                               _bias        = self._weights['Decode_layer_yt_bias'],
                               _activation  = None,
                               _dtype       = tf.float32,
                               _is_bias     = True, )
            (self.decode_cell, self.decode_hidden) = self._LSTMCell(h_t_1 = self.decode_hidden,
                                                                    s_t_1 = self.decode_cell,
                                                                    x_t   = y_t_,
                                                                    name  = 'Decode_LSTMCell')
        pre_y_ = self._Dense(_input       = tf.concat([self.decode_hidden, self.decode_cell], axis=1),
                             _weights     = self._weights['Decode_layer_output_1_weights'],
                             _bias        = self._weights['Decode_layer_output_1_bias'],
                             _activation  = None,
                             _dtype       = tf.float32,
                             _is_bias     = True, )
        pre_y = self._Dense(_input       = pre_y_,
                            _weights     = self._weights['Decode_layer_output_2_weights'],
                            _bias        = self._weights['Decode_layer_output_2_bias'],
                            _activation  = None,
                            _dtype       = tf.float32,
                            _is_bias     = True, )
        return pre_y

    ## 损失函数创建
    def _create_cost_function(self, out):
        # Calculate Cost function:
        self.pred = tf.reshape(out, shape=[-1, ])
        # self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.pred)))
        self.cost = tf.reduce_mean(tf.square(self.y - self.pred))+self.reg_term
        tf.summary.scalar('cost', self.cost)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # optimizer = LookaheadOptimizer(RAdamOptimizer(self.learning_rate))
        self.training_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.LEARN_RATE, self.training_step,  # 10000 * 35091 / 128
                                                        10000, 0.9, staircase=True)  # 10000 represents when _iteration
                                                                                     # ==10000, learning rate starts to
                                                                                     # decay. when staircase=true, not
                                                                                     #　continue to decay.
        tf.summary.scalar('learning_rate', self.learning_rate)
        optimizer = LookaheadOptimizer(tf.train.AdamOptimizer(self.learning_rate), 5, 0.5)
        self.train_op = optimizer.minimize(self.cost)

    ## 打印关键tensor
    def print_tensor(self):
        print('X:{}'.format(self.X))
        print('history_y:{}'.format(self.history_y))
        print('y:{}'.format(self.y))
        print('pred:{}'.format(self.pred))

    ## 训练模型
    def train_model(self,
                    train_x=None,      # shape: [-1, time_step, data_length]
                    train_hist_y=None, # shape: [-1, time_step-1, 1]
                    train_y=None,      # shape: [-1, ]
                    valid_x=None,      # shape: [-1, time_step, data_length]
                    valid_hist_y=None, # shape: [-1, time_step-1, 1]
                    valid_y=None,      # shape: [-1, ]
                    batch_size=None,   # int
                    num_epochs=None,   # the total training steps = (n_samples/batch_size)*num_epochs
                    num_threads=None,  # don't support -1
                    save_name=None, ): # save model name
        if batch_size != self.BATCH_SIZE:
            raise ValueError('batch_size must equal to {}'.format(self.BATCH_SIZE))
        x_batch, y_hist_batch, y_batch = self.get_Batch(data1=train_x,
                                                        data2=train_hist_y,
                                                        label=train_y,
                                                        batch_size=batch_size,
                                                        num_epochs=num_epochs,
                                                        num_threads=num_threads, )
        saver, sess, merged, train_wirter, test_wirter = self.model_init()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        _iteration = 0
        last_cost = 1000
        early_epoch = 0
        try:
            while not coord.should_stop():
                data1, data2, label = sess.run([x_batch, y_hist_batch, y_batch])
                sess.run(self.train_op, feed_dict={self.X: data1, self.history_y: data2, self.y: label})
                _iteration = _iteration + 1
                if _iteration % 200 == 0:
                    train_pre, train_cost, learn_rate, summary = sess.run([self.pred, self.cost, self.learning_rate, merged],
                                                                          feed_dict={self.X: data1, self.history_y: data2,
                                                                                     self.y: label, self.training_step: _iteration})
                    train_wirter.add_summary(summary, _iteration//200)
                    # valid_pre, valid_cost, summary = sess.run([self.pred, self.cost, merged],
                    #                                           feed_dict={self.X: valid_x, self.history_y: valid_hist_y, self.y: valid_y})
                    valid_pre, valid_cost = [], []
                    for i in range(valid_x.shape[0]//batch_size):
                        valid_pre_, valid_cost_, summary = sess.run([self.pred, self.cost, merged],
                                                                    feed_dict={self.X: valid_x[i*batch_size:(i+1)*batch_size, :, :],
                                                                               self.history_y: valid_hist_y[i*batch_size:(i+1)*batch_size, :],
                                                                               self.y: valid_y[i*batch_size:(i+1)*batch_size]})
                        # test_wirter.add_summary(summary, _iteration // 200)
                        valid_pre.append(valid_pre_)
                        valid_cost.append(valid_cost_)
                    test_wirter.add_summary(summary, _iteration // 200)
                    valid_pre = np.array(valid_pre).reshape([-1])
                    valid_cost = np.array(valid_cost).reshape([-1])
                    valid_cost = np.mean(valid_cost)
                    # print("Epoch {}, Train set mse {:.3f}, valid set mse {:.3f}".format(epoch, train_cost, valid_cost))
                    _arr_train_metrict, _arr_valid_metric = [], []
                    _arr_train_metrict = self._metric(label, train_pre, ['MAE', 'RMSE', 'MAPE'])
                    _arr_valid_metrict = self._metric(valid_y, valid_pre, ['MAE', 'RMSE', 'MAPE'])
                    print('Epoch {}: Current learning rate {}'.format(_iteration, round(learn_rate, 7)))
                    print("Epoch {} Train set--MAE:{:.3f} RMSE:{:.3f} MAPE:{:.3%}".format(
                        _iteration, _arr_train_metrict[0], _arr_train_metrict[1], _arr_train_metrict[2]))
                    print("Epoch {} Valid set--MAE:{:.3f} RMSE:{:.3f} MAPE:{:.3%}".format(
                        _iteration, _arr_valid_metrict[0], _arr_valid_metrict[1], _arr_valid_metrict[2]))

                    # if valid_cost < 0.0001:
                    #     saver.save(sess, config.ROOT_PATH + '/_models/DARNN_' + time.strftime +save_name + '/DARNN')
                    #     break
                # if epoch % 10000 == 0:
                #     self.LEARN_RATE = self.LEARN_RATE * (1 - 0.1)
                #     print('Epoch {}: Current learning rate {}'.format(epoch, self.LEARN_RATE))
                    if np.abs(valid_cost - last_cost) < 2:
                        early_epoch += 1
                        if early_epoch > 1000:
                            saver.save(sess, './_models/DARNN_' + save_name + '/DARNN')
                            break
                    else:
                        early_epoch = 0
                    last_cost = valid_cost

        except tf.errors.OutOfRangeError:
            # saver.save(sess, './_models/DARNN_' + save_name + '/DARNN')
            train_wirter.close()
            test_wirter.close()
            print("---Train end---")
        finally:
            coord.request_stop()
            print('---Program end---')
            saver.save(sess, './_models/DARNN_' + save_name + '/DARNN')
            train_wirter.close()
            test_wirter.close()
        coord.join(threads)
    @staticmethod
    ## 模型初始化
    def model_init():
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        train_wirter = tf.summary.FileWriter('./_logs/logs_'+time.strftime("%Y%m%d%H%M")+'/train/', sess.graph)
        test_wirter = tf.summary.FileWriter('./_logs/logs_'+time.strftime("%Y%m%d%H%M")+'/test/')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return saver, sess, merged, train_wirter, test_wirter
    @staticmethod
    ## 得到批次
    def get_Batch(data1=None, data2=None, label=None, batch_size=None, num_epochs=None, num_threads=None):
        input_queue = tf.train.slice_input_producer([data1, data2, label], num_epochs=num_epochs,
                                                    shuffle=True, capacity=32)
        x1_batch, x2_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size,
                                                     num_threads=num_threads, capacity=32,
                                                     allow_smaller_final_batch=False)
        return x1_batch, x2_batch, y_batch
    @staticmethod
    ## 保存tensor到tensorboard
    def variable_summaries(var=None, name=None):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name+'_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    @staticmethod
    ## 定义权重偏置
    def _define_weights(_input_dim  = None,        # network input dimensions (int)
                        _output_dim = None,        # network output dimensions (int)
                        _activation = None,        # activation function (tf.)
                        _dtype      = tf.float32,  # data type (tf.)
                        _name       = None,        # network name (str)
                        _is_bias    = False):      # where to apply bias
        if _activation == tf.nn.relu:
            _r = tf.sqrt(2.0) * tf.sqrt(6.0 / (_input_dim * 1.0 + _output_dim * 1.0))
            _W = tf.Variable(tf.random_uniform([_input_dim, _output_dim], -1. * _r, _r),
                             trainable=True, name=_name + '_W', dtype=_dtype, )
        elif (_activation == tf.nn.softmax) | (_activation == tf.nn.sigmoid):
            _r = tf.sqrt(6.0 / (_input_dim * 1.0 + _output_dim * 1.0))
            _W = tf.Variable(tf.random_uniform([_input_dim, _output_dim], -1. * _r, _r),
                             trainable=True, name=_name + '_W', dtype=_dtype, )
        elif _activation == tf.nn.tanh:
            _r = 4 * tf.sqrt(6.0 / (_input_dim * 1.0 + _output_dim * 1.0))
            _W = tf.Variable(tf.random_uniform([_input_dim, _output_dim], -1. * _r, _r),
                             trainable=True, name=_name + '_W', dtype=_dtype, )
        else:
            _r = tf.sqrt(6.0 / (_input_dim * 1.0 + _output_dim * 1.0))
            _W = tf.Variable(tf.random_uniform([_input_dim, _output_dim], -1. * _r, _r),
                             trainable=True, name=_name + '_W', dtype=_dtype, )
            # _W = tf.Variable(tf.truncated_normal([_input_dim, _output_dim], 0, 1),
            #                  trainable=True, name=_name + '_W', dtype=_dtype, )
        if _is_bias:
            _b = tf.Variable(tf.constant(0., shape=[_output_dim]),
                             trainable=True, name=_name + '_b', dtype=_dtype, )
        else:
            _b = 0
        return (_W, _b)
    @staticmethod
    ## 定义dense
    def _Dense(_is_ensemble = False,       # whether to use tf.layers.dense (bool)
               _input       = None,        # network input (tensor) which not supports multiple dimensions (>2)
               _weights     = None,        # network weights
               _bias        = None,        # network bias
               _output_dim  = None,        # network output dimension
               _activation  = None,        # activation function (tf.)
               _dtype       = tf.float32,  # data type (tf.)
               _name        = None,        # network name (str)
               _is_bias     = False):      # whether to use bias (bool)
        if _is_ensemble:
            _output = tf.layers.dense(_input,
                                      _output_dim,
                                      activation=_activation,
                                      use_bias=_is_bias,
                                      trainable=True,
                                      name=_name,
                                      kernel_initializer=tf.glorot_uniform_initializer(), )
            return _output
        else:
            # y = WT*x+b when x is a vector
            # WT:shape(in_dim, out_dim).transpose()
            # x:shape(in_dim), b:shape(out_dim)
            # _output = tf.matmul(tf.transpose(_W), _input) + _b
            # y = Wx+b when x is not a vector (mini batch)
            # W:shape(in_dim, out_dim)
            # x:shape(-1,in_dim), b:shape(out_dim)
            if _is_bias:
                _output = tf.transpose(tf.tensordot(_weights, _input, axes=[0, 1])) + _bias
            else:
                _output = tf.transpose(tf.tensordot(_weights, _input, axes=[0, 1]))
            if _activation is None:
                return _output
            else:
                return _activation(_output)
    @staticmethod
    ## Attention实现
    def _attention_layer(_h_t_1 = None,     # [-1, cell_dim]
                         _s_t_1 = None,     # [-1, cell_dim]
                         _x_k   = None,     # input attention layer   : [-1, input_dim, time_step]
                                            # temporal attention layer: [-1, time_step, input_dim];
                         _We    = None,     # [2*cell_dim, time_step]
                         _Ue    = None,     # [time_step, time_step]
                         _Ve    = None, ):  # [time_step, 1]
        _input = tf.concat([_h_t_1, _s_t_1], axis=1)  # [-1, 2*cell_dim]
        _output = tf.reshape(tf.transpose(tf.tensordot(_We, _input, axes=[0, 1])), [-1, _We.get_shape().as_list()[-1], 1]) + \
                  tf.transpose(tf.tensordot(_Ue, _x_k, axes=[0, 2]), perm=[1, 0, 2])   # [-1, time_step, input_dim]
        return tf.reshape(tf.tensordot(_Ve, tf.tanh(_output), axes=[0, 1]), \
                          [-1, _x_k.get_shape().as_list()[1]])    # input attention layer   : [-1, input_dim]
                                                                  # temporal attention layer: [-1, time_step]
    @staticmethod
    ## 评估函数
    def _metric(_y_true=None, _y_pred=None, _arr_metric=['MAE']):
        # the shape of _y_true and _y_pred is [-1, ]
        def _rmse():
            tmp = np.mean(np.square(_y_true - _y_pred), keepdims=False)
            return np.round(np.sqrt(tmp),3)
        def _mae():
            tmp = np.mean(np.abs(_y_true - _y_pred), keepdims=False)
            return np.round(tmp,3)
        def _mape():
            tmp = np.mean(np.abs((_y_true-_y_pred)/_y_true), keepdims=False)
            return np.round(tmp, 5)
        _arr_result = []
        for _metric in _arr_metric:
            if _metric == 'MAE':
                _arr_result.append(_mae())
            elif _metric == 'RMSE':
                _arr_result.append(_rmse())
            elif _metric == 'MAPE':
                _arr_result.append(_mape())
            else:
                raise ValueError("metric just only supports 'MAE', 'RMSE' and 'MAPE'")

        return _arr_result


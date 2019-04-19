import tensorflow as tf
from tensorflow import keras
import numpy as np
import decode, encode
import decode2
import generate_data
import bnfunction

import os # python系统库，执行代码，设置虚拟环境。
os.environ['KERAS_BACKEND'] = 'tensorflow'

INPUT_NODE = 1 # 输入节点
OUTPUT_NODE = 2 # 输出节点

LEARNING_RATE_BASE = 0.0001 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 100  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 10000 # 训练多少次

SNR = np.linspace(-10,10,6)# 信噪比



# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='x_input')

y_ = tf.placeholder(tf.float32, [None, int(OUTPUT_NODE * 2)], name='y-input')
y_ber = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input-for-ber')

noise_ = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='noise')


#input_dnn = tf.contrib.layers.batch_norm(tf.reshape(x, [-1, OUTPUT_NODE]), is_training=True)  # reshape + norm
# dnn设计
input_dnn_bn= bnfunction.batchnorm(tf.reshape(x, [-1, OUTPUT_NODE]))


model_dnn_keras_layer1=keras.layers.Dense(8,
                                          activation='relu',
                                          use_bias=True,
                                          activity_regularizer = tf.contrib.layers.l2_regularizer(0.001))(input_dnn_bn)

# mean_layer1,var_layer1 = tf.nn.moments(model_dnn_keras_layer1, axes =[0])
# 进行归一化处理
# model_dnn_keras_layer1_bn=bnfunction.batchnorm(model_dnn_keras_layer1)
model_dnn_keras_layer1_bn=bnfunction.batchnorm(model_dnn_keras_layer1)
model_dnn_keras_layer2=keras.layers.Dense(16,
                                          activation='relu',
                                          use_bias=True,
                                          activity_regularizer=tf.contrib.layers.l2_regularizer(0.001))(model_dnn_keras_layer1_bn)
model_dnn_keras_layer2_bn=bnfunction.batchnorm(model_dnn_keras_layer2)

model_dnn_keras_layer3=keras.layers.Dense(16,
                                          activation='relu',
                                          use_bias=True,
                                          activity_regularizer=tf.contrib.layers.l2_regularizer(0.001))(model_dnn_keras_layer2_bn)
model_dnn_keras_layer3_bn=bnfunction.batchnorm(model_dnn_keras_layer3)


model_dnn_output = keras.layers.Dense(4)(model_dnn_keras_layer3_bn)

norm_value = tf.norm(model_dnn_output, ord='euclidean', axis=None, keep_dims=False, name=None)

model_dnn_output_final = tf.divide(model_dnn_output, norm_value, name=None)
# model_dnn_output_final = tf.norm(model_dnn_output, ord='euclidean', axis=None, keep_dims=False, name=None)

# 交叉熵
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=model_dnn_output))


# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',  # 存储当前迭代的轮数
                              dtype=tf.int32,  # 整数
                              initializer=0,  # 初始化值
                              trainable=False)  # 不可训练
# 定义学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础学习率
                                           global_step,  # 当前迭代轮数
                                           TRAIN_NUM / BATCH_SIZE,  # 迭代次数
                                           LEARNING_RATE_DECAY,  # 学习衰减速度
                                           staircase=False)  # 是否每步都改变速率


# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)


# one-hot 反映射

max_address = tf.argmax(model_dnn_output, 1)

final_value = tf.py_func(decode2.decode, [max_address], tf.float32, name='final_value')

ber = tf.reduce_sum(tf.abs(final_value - y_ber))




# 定义整个模型的x和y

with tf.Session() as sess:
    # tf.global_variables_initializer().run()  # 初始化
    # summary_writer.add_graph(sess.graph)  # 写入变量图
    ber_wrong_number=[]
    # saver = tf.train.Saver(max_to_keep=4)
    saver = tf.train.import_meta_graph('t/modulate-model-900.meta')
    # x = tf.get_default_graph().get_operation_by_name('x_input').outputs[0]
    # final_value = tf.get_default_graph().get_operation_by_name('final_value').outputs[0]
    # y_ber = tf.get_default_graph().get_operation_by_name('y-input-for-ber').outputs[0]
    # ber = tf.reduce_mean(tf.abs(final_value - y_ber))

    for i in range(len(SNR)):
        Y = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])  # 让数据的0和1的概率都相同
        X = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2
        Y_one_hot = encode.encode2d_onehot(Y)  # 单热码用于验证数据
        E_x = 10 ** (0.1 * SNR[i])  # 信号能量
        noise = np.random.randn(TRAIN_NUM, INPUT_NODE, 2)  # sigma * r + mu
        X = X * E_x + noise
        # saver=tf.train.import_meta_graph('t/modulate-model-900.meta')
        saver.restore(sess, tf.train.latest_checkpoint('t/'))

        ber_loss = sess.run(ber,
                            feed_dict={x: X, y_ber: Y})

        ber_wrong_number.append((ber_loss) / (TRAIN_NUM * OUTPUT_NODE))

        all_wrong = np.sum(ber_wrong_number, axis=0)    #axis=1 是按行求和



        print('ber为%f' % ( ber_wrong_number[i] ))
        # saver.save(sess,'t/modulate-model',global_step=i)
sess.close()

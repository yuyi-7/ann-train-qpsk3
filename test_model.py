import tensorflow as tf
from tensorflow import keras
import numpy as np
import decode, encode
import decode2
import generate_data

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

with tf.Session() as sess:
    # tf.global_variables_initializer().run()  # 初始化
    # summary_writer.add_graph(sess.graph)  # 写入变量图
    ber_wrong_number=[]
    # saver = tf.train.Saver(max_to_keep=4)
    saver = tf.train.import_meta_graph('t/modulate-model-900.meta')
    x = tf.get_default_graph().get_operation_by_name('x_input').outputs[0]
    final_value = tf.get_default_graph().get_operation_by_name('final_value').outputs[0]
    y_ber = tf.get_default_graph().get_operation_by_name('y-input-for-ber').outputs[0]
    ber = tf.reduce_mean(tf.abs(final_value - y_ber))

    for i in SNR:
        Y = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])  # 让数据的0和1的概率都相同
        X = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2
        Y_one_hot = encode.encode2d_onehot(Y)  # 单热码用于验证数据
        E_x = 10 ** (0.1 * i)  # 信号能量
        noise = np.random.randn(TRAIN_NUM, INPUT_NODE, 2)  # sigma * r + mu
        X = X * E_x + noise
        # saver=tf.train.import_meta_graph('t/modulate-model-900.meta')
        saver.restore(sess, tf.train.latest_checkpoint('t/'))

        ber_loss = sess.run(ber,
                            feed_dict={x: X, y_ber: Y})

        ber_wrong_number.append(ber_loss)

        all_wrong = np.sum(ber_wrong_number, axis=0)    #axis=1 是按行求和



        print('ber为%f' % ( ber_loss ))
        # saver.save(sess,'t/modulate-model',global_step=i)
sess.close()

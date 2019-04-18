import tensorflow as tf
from tensorflow import keras
import numpy as np
import decode, encode
import decode2
import generate_data
import bnfunction
from sklearn.preprocessing import OneHotEncoder as hot1
import os # python系统库，执行代码，设置虚拟环境。
import time

os.environ['KERAS_BACKEND'] = 'tensorflow'

INPUT_NODE = 1 # 输入节点
OUTPUT_NODE = 2 # 输出节点

LEARNING_RATE_BASE = 0.0001 # 模型基础学习速率
LEARNING_RATE_DECAY = 0.99  # 学习衰减速度
BATCH_SIZE = 100  # 一批数据量
TRAIN_NUM = 20000  # 数据总量
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减
TRAINING_STEPS = 1000 # 训练多少次

SNR = 5 # 信噪比

E_x = 10 ** (0.1*SNR)  # 信号能量

# 生成数据
# Y = np.random.randint(0,2,[TRAIN_NUM , OUTPUT_NODE]).astype('float32')
Y = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])  # 让数据的0和1的概率都相同

X = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2
Y_one_hot=encode.encode2d_onehot(Y) # 单热码用于验证数据

# 验证数据测试集
Y_vaildate = generate_data.generate_bit([TRAIN_NUM, OUTPUT_NODE])

X_validate = encode.encode2d(Y)  # TRAIN_NUM , OUTPUT_NODE / 2 , 2

Y_validate_one_hot= encode.encode2d_onehot(Y_vaildate) # 单热码用于验证数据


# 加噪声
noise = np.random.randn(TRAIN_NUM, INPUT_NODE, 2)  # sigma * r + mu
X = X * E_x + noise

# 定义整个模型的x和y
x = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='x_input')

y_ = tf.placeholder(tf.float32, [None, int(OUTPUT_NODE * 2)], name='y-input')
y_ber = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input-for-ber')

noise_ = tf.placeholder(tf.float32, [None, INPUT_NODE, 2], name='noise')


input_dnn = tf.contrib.layers.batch_norm(tf.reshape(x, [-1, OUTPUT_NODE]), is_training=True)  # reshape + norm
# dnn设计
input_dnn_bn=bnfunction.batchnorm(input_dnn)


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

# 交叉熵loss
# model_dnn_output = model_dnn_output / tf.norm(model_dnn_output, 2, axis=0)
# model_output_sigmoid = tf.nn.sigmoid(model_dnn_output)
# cross_entroy = -tf.add(model_dnn_output * tf.log(y_),
#                        tf.subtract(1., model_dnn_output) * tf.log(tf.subtract(1. , y_)))  # -[y*ln(y_) + (1-y)*ln(1-y_)]
# loss = tf.reduce_mean(cross_entroy)

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

summary_writer = tf.summary.FileWriter('logs/dnn_log')  # tensorboard保存目录，目录以时间命名


# 启动
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 初始化
    summary_writer.add_graph(sess.graph)  # 写入变量图
    ber_wrong_number=[]
    saver = tf.train.Saver(max_to_keep=4)
    for i in range(TRAINING_STEPS):
        # 设置批次
        start = (i * BATCH_SIZE) % TRAIN_NUM
        end = min(start + BATCH_SIZE, TRAIN_NUM)

        batch_num = end - start

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_one_hot[start:end]})

        ber_loss = sess.run(ber,
                            feed_dict={x: X[start:end], y_ber: Y[start:end]})

        ber_wrong_number.append(ber_loss)
        compute_loss = sess.run(loss,
                                feed_dict={x: X[start:end], y_: Y_one_hot[start:end]})

        # model_output_sigmoid_ = sess.run(model_output_sigmoid,
        #                                  feed_dict={x: X[start:end], y_: Y_one_hot[start:end]})

        validate_loss = sess.run(loss,
                                 feed_dict={x: X_validate[start:end],
                                            y_: Y_validate_one_hot[start:end],
                                            })
        all_wrong = np.sum(ber_wrong_number, axis=0)    #axis=1 是按行求和


        # 输出
        if i % 100 == 0:
            print('训练了%d次,总损失%f,ber为%f,验证损失%f' % (i, compute_loss, (all_wrong)/(i*batch_num*2), validate_loss))
            saver.save(sess,'t/modulate-model',global_step=i)
        # if (i % 200 == 0) and (i != 0):
            # print(model_output_sigmoid_)
            # print('模型预测结果:', sess.run(y, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            # print('实际结果:', sess.run(y_, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            # print('加上噪声:', sess.run(x, feed_dict={x: X[start:start + 1], y_: Y[start:start + 1]}))
            # print('批归一化:', sess.run(input_cnn, feed_dict={x: X[start:start+1], y_: Y[start:start+1]}))
            # # print('DNN后:', sess.run(y, feed_dict={x: X[start:end], y_: Y[start:end]}))
            # print('weight:', sess.run(weight, feed_dict={x: X[start:end], y_: Y[start:end]}))
sess.close()

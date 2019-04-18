import tensorflow as tf


def batchnorm(y):
    # x=[]
    mean_layer, var_layer = tf.nn.moments(y, axes=[0])
    a=tf.subtract(y, mean_layer, name=None)   #
    b=tf.sqrt(var_layer, name=None)          # 开根号，必须传入浮点数或复数
    norm_value=tf.divide(a, b, name=None)     # 浮点除法, 返回浮点数(python3 除法)
    return norm_value




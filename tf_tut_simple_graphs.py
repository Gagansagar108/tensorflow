import tensorflow as tf
import numpy as np


w=tf.Variable(2.0,tf.float32)
b=tf.Variable(2.8,tf.float32)
X=tf.placeholder(tf.float32)
line=X*w+b

init=tf.global_variables_initializer()
sess=tf.Session()

sess.run(init)
print(sess.run(line,{X:[1,2,3,34,5]}))

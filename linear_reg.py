import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
X_train=np.linspace(-1,1,100)
y_train=3*X_train+np.random.randn(100)

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
w=tf.Variable(0.0)

y_model=X*w

cost=tf.square(Y-y_model)
train_op=tf.train.ProximalGradientDescentOptimizer(0.01).minimize(cost)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for x in range(5):
    sess.run(train_op,{X:X_train,Y:y_train})
    print(sess.run(w))
    continue

































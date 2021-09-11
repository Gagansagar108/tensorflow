import tensorflow as tf
import numpy as np


hello=tf.constant("hello_tf")


c1=tf.constant(5)
c2=tf.constant(6)

sess=tf.Session()
print(sess.run(hello))

print(sess.run([c1,c2]))
tf_t=tf.convert_to_tensor(5)
print(tf_t)


array_2d=np.array([[1,2,3,4],[3,6,7,8]])
print("array_2d_np:" ,array_2d)
array_2d_tf=tf.convert_to_tensor(array_2d)
print(sess.run(array_2d_tf))


a=tf.zeros((2,3))
print("zeros:--",sess.run(a))

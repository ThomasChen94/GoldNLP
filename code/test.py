import tensorflow as tf

a = tf.constant([[1.0,2,3], [1,2,3]])
b = tf.constant([[4,5,6], [4,5,6]])


sess = tf.Session()
print(sess.run(a / a))


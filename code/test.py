import tensorflow as tf

a = tf.constant([[[1.0,2,3], 
				   [1,2,3]], 

				  [[3,4,5],
				   [3,4,5]]])

b = tf.constant([[4,5,6], [4,5,6]])


sess = tf.Session()
print(sess.run(tf.reduce_sum(a, axis = 1)))


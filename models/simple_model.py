import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello TensorFlow from Salpe")

print(sess.run(hello))
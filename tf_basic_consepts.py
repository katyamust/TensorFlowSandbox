
tensor_r0 = 3 # a rank 0 tensor; this is a scalar with shape []
tensor_r1 = [1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
tensor_r2 = [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
tensor_r3 = [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

#placeholder/vairable
#node
#session

import tensorflow as tf

# tf place holder a can be scalar and vector
a = tf.placeholder(tf.float32)

a_var = tf.Variable([.3], tf.float32)

b = tf.placeholder(tf.float32)

# declare operation computational node
adder = tf.add(a,b)  # provides a shortcut for tf.add(a,b)
print(a)
print(a_var)
print(b)

#run in session
sess = tf.Session()
res = sess.run(adder, {a: 12, b:13})
print(res)

print("End of boring stuff")
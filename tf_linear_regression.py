import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

# loss
loss = tf.reduce_mean(tf.square(linear_model - y))# sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# training data
x_train = np.random.rand(100).astype(np.float32)
y_noise = np.random.ranf([100])
y_train = x_train * 3 + 4 #+ y_noise

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong
for i in range(100):
    sess.run(train, {x:x_train, y:y_train})
    if i % 10 == 0:
        print(i, sess.run(W),sess.run(b))

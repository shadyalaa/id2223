import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import reader as rd

NUM_HIDDEN_NODES = 50
NUM_EPOCHS = 1500

X = tf.placeholder(tf.float32, [None, 26], name="X")
Y = tf.placeholder(tf.float32, [None, 26], name="Y")
Z = tf.placeholder(tf.float32, [None, 26], name="Z")

def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
	if init_method == 'zeros':
		return tf.Variable(tf.zeros(shape, dtype=tf.float32))
	elif init_method == 'uniform':
		return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
	else: #xavier
		(fan_in, fan_out) = xavier_params
		low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
		high = 4*np.sqrt(6.0/(fan_in + fan_out))
		return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def model(X, Y, num_hidden=10):
    wx_h = init_weights([26, num_hidden], 'xavier', xavier_params=(26, num_hidden))
    wy_h = init_weights([26, num_hidden], 'xavier', xavier_params=(26, num_hidden))
    b_h = init_weights([num_hidden], 'zeros')
    h = tf.nn.sigmoid(tf.matmul(X, wx_h) + tf.matmul(Y, wy_h) + b_h)
    w_o = init_weights([num_hidden, 26], 'xavier', xavier_params=(num_hidden, 26))
    b_o = init_weights([26], 'zeros')
    z =  tf.matmul(h, w_o) + b_o
    return z


zhat = model(X, Y, NUM_HIDDEN_NODES)


#train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(tf.div(tf.reduce_mean(tf.squared_difference(zhat,Z)),tf.reduce_mean(tf.square(Z)))))
#train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(tf.reduce_mean(tf.div(tf.squared_difference(zhat,Z),tf.square(Z)))))
#error = tf.div(2*tf.abs(tf.sub(zhat,Z)),tf.add(zhat,Z))
#loss = tf.reduce_sum(error)
loss = tf.reduce_sum(tf.squared_difference(zhat,Z))
train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(loss))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

errors = []
for i in range(NUM_EPOCHS):
    T,q,lwhr = rd.nextBatch(100,i)
    _ , mse = sess.run((train_op,loss), feed_dict={X: T, Y: q, Z: lwhr})
    if i!=0 and i%100 == 0:
	T,q,lwhr = rd.validBatch()
	_,pred = sess.run((tf.nn.l2_loss(tf.reduce_mean(tf.squared_difference(zhat,lwhr))),zhat),  feed_dict={X:T, Y:q})
        #pred = sess.run(zhat,  feed_dict={X:T, Y:q})
	#mse = sess.run(tf.nn.l2_loss(tf.reduce_mean(tf.div(tf.squared_difference(zhat,lwhr),tf.square(lwhr)))),  feed_dict={X:T, Y:q})
	#errors.append(mse) 
	print "epoch %d, validation MSE %g" % (i, mse)
	out.close()

plt.figure(1)

plt.plot(lwhr[0])
plt.plot(pred[0])
plt.xlabel('index')
plt.ylabel('out')

plt.figure(2)

plt.plot(lwhr[10])
plt.plot(pred[10])
plt.xlabel('index')
plt.ylabel('out')


plt.show()

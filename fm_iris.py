import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# data load
iris = load_iris()
x = iris["data"]
y = iris["target"]

x, y = x[y != 2], y[y != 2]
for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9415)
n = 4
k = 4

# write fm algorithm
w0 = tf.Variable(0.1)
w1 = tf.Variable(tf.truncated_normal([n]))
w2 = tf.Variable(tf.truncated_normal([n, k]))

x_ = tf.placeholder(tf.float32, [None, n])
y_ = tf.placeholder(tf.float32, [None])
batch = tf.placeholder(tf.int32)

w2_new = tf.reshape(tf.tile(w2, [batch, 1]), [-1, 4, k])
board_x = tf.reshape(tf.tile(x_, [1, k]), [-1, 4, k])
board_x2 = tf.square(board_x)

q = tf.square(tf.reduce_sum(tf.multiply(w2_new, board_x), axis=1))
h = tf.reduce_sum(tf.multiply(tf.square(w2_new), board_x), axis=1)

y_fm = w0 + tf.reduce_sum(tf.multiply(x_, w1), axis=1) + \
       1 / 2 * tf.reduce_sum(q - h, axis=1)

cost = tf.reduce_sum(0.5 * tf.square(y_fm - y_))+tf.contrib.layers.l2_regularizer(0.1)(w0) \
    +tf.contrib.layers.l2_regularizer(0.1)(w1) + tf.contrib.layers.l2_regularizer(0.1)(w2)
batch_fl = tf.cast(batch, tf.float32)
accury = (batch_fl + tf.reduce_sum(tf.sign(tf.multiply(y_fm, y_)))) / (batch_fl * 2)
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_op, feed_dict={x_: x_train, y_: y_train, batch: 70})
        print()
        sess.run(cost, feed_dict={x_: x_train, y_: y_train, batch: 70})
        # sess.run(accury, feed_dict={x_: x_test, y_: y_test, batch: 30})
        # print('test_acccuracy' + str(accury))
    print()
    sess.run(accury, feed_dict={x_: x_test, y_: y_test, batch: 30})
    print('test_acccuracy'+str(accury))
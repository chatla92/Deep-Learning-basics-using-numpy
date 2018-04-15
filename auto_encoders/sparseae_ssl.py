import tensorflow as tf
import numpy as np
from load_dataset import mnist
from sklearn.utils import shuffle

np.random.seed(0)

learning_rate = 0.1
iterations = 400
lambd = 0.0001
beta = 3.0

input_x_labeled = []
input_y_labeled = []
input_x_unlabeled = []
input_y_unlabeled = []

for i in range(1, 11):
    temp_data, temp_label, _, _ = mnist(ntrain=60000, ntest=1, digit_range=[i - 1, i])
    input_x_labeled.extend(temp_data.T[0:100])
    input_y_labeled.extend(temp_label.T[0:100])
    input_x_unlabeled.extend(temp_data.T[100:])
    input_y_unlabeled.extend(temp_label.T[100:])

input_x_labeled, input_y_labeled = shuffle(input_x_labeled, input_y_labeled, random_state=10)
input_x_unlabeled, input_y_unlabeled = shuffle(input_x_unlabeled, input_y_unlabeled, random_state=10)

input_x_labeled = np.array(input_x_labeled).T
input_y_labeled = np.array(input_y_labeled).T
input_x_unlabeled = np.array(input_x_unlabeled).T
input_y_unlabeled = np.array(input_y_unlabeled).T

# Train the Auto encoder
num_of_hidden_units = 200
input_size = 784
output_classes = 10
rho = 0.1

X_ae = tf.placeholder(tf.float32, [input_size, None])

weights = dict()
biases = dict()

weights["en"] = tf.Variable(tf.random_normal([num_of_hidden_units, input_size]))#tf.Variable(tf.truncated_normal([num_of_hidden_units, input_size], stddev=0.001))
biases["en"] = tf.Variable(tf.zeros([num_of_hidden_units, 1]))
weights["de"] = tf.Variable(tf.random_normal([input_size, num_of_hidden_units]))#tf.Variable(tf.truncated_normal([input_size, num_of_hidden_units], stddev=0.001))
biases["de"] = tf.Variable(tf.zeros([input_size, 1]))

z_encoded = tf.add(tf.matmul(weights["en"], X_ae), biases["en"])
encoded = tf.nn.sigmoid(z_encoded)

z_decoded = tf.add(tf.matmul(weights["de"], encoded), biases["de"])
decoded = tf.nn.sigmoid(z_decoded)

avg_act = tf.reduce_mean(encoded, axis=1)

fro_en = tf.sqrt(tf.nn.l2_loss(tf.reshape(weights["en"], [-1])))
fro_de = tf.sqrt(tf.nn.l2_loss(tf.reshape(weights["de"], [-1])))
kl = tf.multiply(rho, tf.log(rho)) - tf.multiply(rho, tf.log(avg_act)) + tf.multiply((1-rho), tf.log(1-rho)) - tf.multiply((1-rho), tf.log(1-avg_act))
loss = tf.reduce_mean(tf.square(X_ae - decoded)) + (lambd*0.5*(fro_en + fro_de)) + (beta*tf.reduce_mean(kl))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for ii in range(iterations):
        _, cost = sess.run([optimizer, loss], feed_dict={X_ae: input_x_labeled})

        if ii % 100 == 0:
            print("Autoencoder Cost at iteration %i is: %.05f" % (ii, cost))

    softmax_input = sess.run(encoded, feed_dict={X_ae: input_x_labeled})

# Train Softmax classifier
X_hidden = tf.placeholder(tf.float32, [num_of_hidden_units, None])
Y_hidden = tf.placeholder(tf.float32, [output_classes, None])

weights["hi"] = tf.Variable(tf.random_normal([output_classes, num_of_hidden_units], stddev=0.01))
biases["hi"] = tf.Variable(tf.zeros([output_classes, 1]))

one_hot_matrix_labeled = tf.one_hot(input_y_labeled, 10, axis=0)
one_hot_matrix_unlabeled = tf.one_hot(input_y_unlabeled, 10, axis=0)

Z = tf.add(tf.matmul(weights["hi"], X_hidden), biases["hi"])

logits = tf.transpose(Z)
labels = tf.transpose(Y_hidden)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    one_hot_matrix_labeled = sess.run(one_hot_matrix_labeled)[:, 0, :]
    for ii in range(iterations):
        _, cost = sess.run([optimizer, loss], feed_dict={X_hidden: softmax_input, Y_hidden: one_hot_matrix_labeled})

        if ii % 100 == 0:
            print("Softmax Classifier Cost at iteration %i is: %.05f" % (ii, cost))

# Fine tune  SSL
X = tf.placeholder(tf.float32, [input_size, None])
Y = tf.placeholder(tf.float32, [output_classes, None])

Z1 = tf.add(tf.matmul(weights["en"], X), biases["en"])
A1 = tf.nn.sigmoid(Z1)
Z2 = tf.add(tf.matmul(weights["hi"], A1), biases["hi"])

logits = tf.transpose(Z2)
labels = tf.transpose(Y)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    one_hot_matrix_unlabeled = sess.run(one_hot_matrix_unlabeled)[:, 0, :]
    preds = tf.equal(tf.argmax(Z2), tf.argmax(Y))
    acc = tf.reduce_mean(tf.cast(preds, "float"))
    for ii in range(iterations):
        _, cost = sess.run([optimizer, loss], feed_dict={X: input_x_labeled, Y: one_hot_matrix_labeled})

        if ii % 100 == 0:
            print("SSL finetuning Cost at iteration %i is: %.05f" % (ii, cost))
    print ("Train Accuracy:", acc.eval({X: input_x_labeled, Y: one_hot_matrix_labeled}))
    print ("Test Accuracy:", acc.eval({X: input_x_unlabeled, Y: one_hot_matrix_unlabeled}))

# Train plain Neural Network
X_nn = tf.placeholder(tf.float32, [input_size, None])
Y_nn = tf.placeholder(tf.float32, [output_classes, None])

w1 = tf.Variable(tf.truncated_normal([num_of_hidden_units, input_size]))
b1 = tf.Variable(tf.zeros([num_of_hidden_units, 1]))
w2 = tf.Variable(tf.truncated_normal([output_classes, num_of_hidden_units]))
b2 = tf.Variable(tf.zeros([output_classes, 1]))

Z1 = tf.add(tf.matmul(w1, X_nn), b1)
A1 = tf.nn.sigmoid(Z1)
Z2 = tf.add(tf.matmul(w2, A1), b2)

logits = tf.transpose(Z2)
labels = tf.transpose(Y_nn)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    preds = tf.equal(tf.argmax(Z2), tf.argmax(Y_nn))
    acc = tf.reduce_mean(tf.cast(preds, "float"))
    for ii in range(iterations):
        _, cost = sess.run([optimizer, loss], feed_dict={X_nn: input_x_labeled, Y_nn: one_hot_matrix_labeled})

        if ii % 100 == 0:
            print("Plain NN Cost at iteration %i is: %.05f" % (ii, cost))
    print ("Train Accuracy:", acc.eval({X_nn: input_x_labeled, Y_nn: one_hot_matrix_labeled}))
    print ("Test Accuracy:", acc.eval({X_nn: input_x_unlabeled, Y_nn: one_hot_matrix_unlabeled}))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import mnist

learning_rate = 0.1
iterations = 400
sparsity = [0.01, 0.1, 0.5, 0.8]
lambd = 0.0001
beta = 3.0

num_of_hidden_units = 200
input_size = 784

input_x, _, _, _ = mnist(ntrain=60000, ntest=10, digit_range=[0, 10], shuffle=True)

fig_n = 1
for r in sparsity:
    print("For sparsity parameter : " + str(r))

    X = tf.placeholder(tf.float32, [input_size, None])
    rho = tf.placeholder(tf.float32)

    weights = dict()
    biases = dict()
    
    weights["en"] = tf.Variable(tf.random_normal([num_of_hidden_units, input_size], stddev=0.01))#tf.Variable(tf.truncated_normal([num_of_hidden_units, input_size], stddev=0.001))
    biases["en"] = tf.Variable(tf.zeros([num_of_hidden_units, 1]))
    weights["de"] = tf.Variable(tf.random_normal([input_size, num_of_hidden_units], stddev=0.01))#tf.Variable(tf.truncated_normal([input_size, num_of_hidden_units], stddev=0.001))
    biases["de"] = tf.Variable(tf.zeros([input_size, 1]))
    
    z_encoded = tf.add(tf.matmul(weights["en"], X), biases["en"])
    encoded = tf.nn.sigmoid(z_encoded)

    z_decoded = tf.add(tf.matmul(weights["de"], encoded), biases["de"])
    decoded = tf.nn.sigmoid(z_decoded) 
    
    avg_act = tf.reduce_mean(encoded, axis=1)

    fro_en = tf.nn.l2_loss(weights["en"])#tf.multiply(lambd, tf.reduce_sum(tf.pow(tf.reshape(weights["en"], [-1]), 2)) )
    fro_de = tf.nn.l2_loss(weights["de"])#tf.multiply(lambd, tf.reduce_sum(tf.pow(tf.reshape(weights["de"], [-1]), 2)) )
    kl = tf.multiply(rho, tf.log(rho)) - tf.multiply(rho, tf.log(avg_act)) + tf.multiply((1-rho), tf.log(1-rho)) - tf.multiply((1-rho), tf.log(1-avg_act))
    
    loss = tf.reduce_mean(tf.square(X - decoded)) + (lambd*0.5*(fro_en + fro_de)) + (beta*tf.reduce_mean(kl))
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        
        for ii in range(iterations):
            _, cost = sess.run([optimizer, loss], feed_dict={X: input_x, rho:r})

            if ii % 100 == 0:
                print("Cost at iteration %i is: %.05f" %(ii, cost))

        fig = plt.figure()
        w = sess.run(weights["en"])
        for num in range(100):

            y = fig.add_subplot(10, 10, num + 1)
            orig = w[num, :].reshape(28, 28)
            
            y.imshow(orig, cmap='gray')
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

        plt.savefig('rho'+str(fig_n)+'.png', dpi=1000)
        
        fig = plt.figure()
        input_x, _, _, _ = mnist(ntrain=100, ntest=1, digit_range=[0, 10]) 
        w = sess.run(decoded,feed_dict={X: input_x})
        w = np.transpose(w)
        for num, data in enumerate(w[:100]):

            y = fig.add_subplot(10, 10, num + 1)
            orig = data.reshape(28, 28)

            y.imshow(orig, cmap='gray')
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)

        plt.savefig('rec_rho'+str(fig_n)+'.png', dpi=1000)
        fig_n +=1
    tf.reset_default_graph()

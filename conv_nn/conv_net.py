from load_dataset import mnist
from helper_im2col import *
import matplotlib.pyplot as plt


def classify(X, parameters):

    A, _ = conv_layer_forward(X, parameters)

    if parameters['pool']:
        A, cache2 = pool_forward(A, parameters)

    (m, Ht, Wd, ch) = A.shape
    A = A.reshape(m, Ht*Wd*ch)
    A = A.T
    A, _ = layer_forward(A, parameters['W2'], parameters['b2'], "relu")

    if parameters['dropout']:
        A, cache_d = dropout_forward(A, parameters['dropout_prob'])

    A, _ = layer_forward(A, parameters['W3'], parameters['b3'], "linear")
    A_fin, _, _ = softmax_cross_entropy_loss(A)
    Ypred = np.argmax(A_fin, axis=0).astype(float)

    return Ypred


def conv_net(X, Y, net_dims, num_iterations=2000, learning_rate=0.1):
    no_of_filters = 5
    filter_size = 3

    pad = 1
    stride = 1

    pool = False
    pool_size = 2
    pool_stride = 2

    dropout = False
    dropout_prob = 0.5

    input_dims, n_h, n_fin = net_dims
    (m, Ht_input, Wd_input, Ch_input) = input_dims

    parameters = dict()
    parameters['W1'] = np.random.randn(filter_size, filter_size, Ch_input, no_of_filters) * np.sqrt(2/3)
    parameters['b1'] = np.zeros((1, 1, 1, no_of_filters), dtype='float32')

    parameters['W2'] = np.random.randn(n_h, (Ht_input * Wd_input * no_of_filters)) * np.sqrt(2/(Ht_input * Wd_input * no_of_filters))

    if pool:
        parameters['W2'] = np.random.random((n_h, (int(1 + (Ht_input - pool_size) / pool_stride) * int(1 + (Wd_input - pool_size) / pool_stride) * no_of_filters))) * 0.01

    parameters['b2'] = np.zeros((n_h, 1), dtype='float32')

    parameters['W3'] = np.random.randn(n_fin, n_h) * np.sqrt(2/n_h)
    parameters['b3'] = np.zeros((n_fin, 1), dtype='float32')

    parameters['pad'] = pad

    parameters['stride'] = stride
    parameters['filter_size'] = filter_size
    parameters['no_of_filters'] = no_of_filters

    parameters['pool_size'] = pool_size
    parameters['pool_stride'] = pool_stride
    parameters['pool'] = pool

    parameters['dropout'] = dropout
    parameters['dropout_prob'] = dropout_prob

    A0 = X
    costs = []
    for ii in range(num_iterations):
        alpha = learning_rate * (1 / (1 + 0.01 * ii))
        print(alpha)
        # Forward Propagation
        A, cache1 = conv_layer_forward(A0, parameters)
        A, cache_r = relu(A)

        if pool:
            A, cache2 = pool_forward(A, parameters)
        (m, Ht_input, Wd_input, no_of_filters) = A.shape
        A = A.reshape(m, Ht_input * Wd_input * no_of_filters)
        A = A.T

        A, cache3 = layer_forward(A, parameters['W2'], parameters['b2'], "relu")

        if dropout:
            A, cache_d = dropout_forward(A, dropout_prob)

        A, cache4 = layer_forward(A, parameters['W3'], parameters['b3'], "linear")

        A_fin, cache, cost = softmax_cross_entropy_loss(A, Y)
        dZ = softmax_cross_entropy_loss_der(Y, cache)

        # Backward Propagation
        dA, dW3, db3 = layer_backward(dZ, cache4, parameters['W3'], parameters['b3'], "linear")

        if dropout:
            dA = dropout_backward(dA, cache_d)

        dA, dW2, db2 = layer_backward(dA, cache3, parameters['W2'], parameters['b2'], "relu")
        dA = dA.T

        dA = dA.reshape(m, Ht_input, Wd_input, no_of_filters)

        if pool:
            dA = pool_backward(dA, cache2)

        dA = relu_der(dA, cache_r)
        dA, dW1, db1 = conv_layer_backward(dA, cache1)

        parameters['W1'] = parameters['W1'] - (alpha * dW1)
        parameters['b1'] = parameters['b1'] - (alpha * db1)
        parameters['W2'] = parameters['W2'] - (alpha * dW2)
        parameters['b2'] = parameters['b2'] - (alpha * db2)
        parameters['W3'] = parameters['W3'] - (alpha * dW3)
        parameters['b3'] = parameters['b3'] - (alpha * db3)

        # if ii % 10 == 0:
        #     costs.append(cost)
        # if ii % 100 == 0:
        #     print("Cost at iteration %i is: %f" % (ii, cost))
        costs.append(cost)
        print("Cost at iteration %i is: %f" % (ii, cost))
    return costs, parameters


def main():
    train_data, train_label, test_data, test_label = mnist(ntrain=50, ntest=10, digit_range=[0, 10])
    (m, H_in, W_in, Ch_in) = train_data.shape
    n_fin = 10
    n_h = 100

    net_dims = [(m, H_in, W_in, Ch_in), n_h, n_fin]

    learning_rate = 0.2
    num_iterations = 100

    costs, parameters = conv_net(train_data, train_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)

    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    trAcc = np.sum(train_Pred == train_label) / float(train_label.shape[1]) * 100
    teAcc = np.sum(test_Pred == test_label) / float(test_label.shape[1]) * 100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

    plt.plot(range(1, len(costs) + 1), costs, label='Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Mini Project3:1.2 Convolutional neural network with Max Pooling (Two-layer)-Cost vs Iterations')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

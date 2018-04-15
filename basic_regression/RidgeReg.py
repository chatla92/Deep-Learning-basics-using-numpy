'''
Author: Anjani Sai Kumar Chatla
ASU ID: 1211029375
Email ID: achatla@asu.edu
Python Version: 2.7
'''
#!/usr/bin/env python2.7
import numpy
# if there is an issue with matplotlib on your computer run echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
import matplotlib.pyplot as plt
import itertools

# Load Data
data = numpy.load('linRegData.npy')
# Randomize Input data
numpy.random.shuffle(data)
data = numpy.matrix(data)

# K value for cross fold validation
cv_fold = 10

# Get X and Y
m = data.shape[0]
x1 = data[:, 0]
x = numpy.append(numpy.ones((m, 1),  dtype=x1.dtype), x1, axis=1)

# Append Ridge Regression polynomials (x^2...x^15) to input
for i in range(2, 16):
    x = numpy.append(x, numpy.power(x1, i), axis=1)

y = data[:, 1]


lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 5, 10]

# Placeholders for Validation and Training Errors
error_val = [0]*len(lambdas)
error_train = [0]*len(lambdas)

for i, lam in enumerate(lambdas):

    reg_term = numpy.diag(numpy.array([(lam * lam) for _ in range(16)]))

    # Calculate Validation and Training Error for
    for k in range(cv_fold):

        # Start index for validation set
        test_ind = k*(m/cv_fold)

        # Validation set input data :x_test, Validation set true label:y_test
        x_test = x[test_ind:test_ind+(m/cv_fold)]
        y_test_true = y[test_ind:test_ind + (m / cv_fold)]

        # Training set input data :x_train, Training set true label:y_train_true
        x_train = numpy.vstack((numpy.array(x[0:test_ind]), numpy.array(x[test_ind+(m/cv_fold):])))
        y_train_true = numpy.vstack((numpy.array(y[0:test_ind]), numpy.array(y[test_ind + (m / cv_fold):])))


        # Calculate ((X^T*X) + I*lambda^2)^-1
        temp1 = numpy.add(x_train.transpose().dot(x_train), reg_term)
        temp1 = numpy.asmatrix(temp1).I

        # Calculate (X^T*Y)
        temp2 = x_train.transpose().dot(y_train_true)

        # Calculate  theta = (((X^T*X) + I*lambda^2)^-1*((X^T*Y))
        theta = temp1.dot(temp2)

        # Predictions for Training set and validation set
        y_train = x_train.dot(theta)
        y_test = x_test.dot(theta)

        # For a given Lambda, Sum of cross validation errors across different folds
        error_train[i] = error_train[i] + (numpy.power((y_train_true-y_train), 2).sum())/(m-m/cv_fold)
        error_val[i] = error_val[i] + (numpy.power((y_test_true-y_test), 2).sum())/(m/cv_fold)

# Calculate Average of Training and Validation Errors
for i in range(len(lambdas)):
    error_train[i] = error_train[i]/cv_fold
    error_val[i] = error_val[i]/cv_fold

print "Training Error: "+ str(error_train)
print "Validation Error: "+ str(error_val)

# Select the Lambda with minimum cross validation error
min_error = min(error_val)
l = lambdas[error_val.index(min_error)]
print "Lambda with least cross validation error is: " + str(l) + " with error: " + str(min_error)

# polynomial fit of the input data
reg_term_final = numpy.diag(numpy.array([(l * l) for _ in range(16)]))

temp1_final = numpy.add(x.transpose().dot(x), reg_term_final)
temp1_final = numpy.asmatrix(temp1_final).I

# Calculate (X^T*Y)
temp2_final = x.transpose().dot(y)

# Calculate  theta = (((X^T*X) + I*lambda^2)^-1*((X^T*Y))
theta_final = temp1_final*temp2_final

y_prime = x.dot(theta_final)

lists = sorted(itertools.izip(*[numpy.squeeze(numpy.asarray(x1)), numpy.squeeze(numpy.asarray(y_prime))]))
new_x1, new_y_prime = list(itertools.izip(*lists))

plt.figure(1)
plt.plot(numpy.log(lambdas), error_train, marker='o', label='Training Error')
plt.plot(numpy.log(lambdas), error_val, marker='o', linestyle='--', label='Evaluation Error')
plt.xlabel('Lambda')
plt.ylabel('Training Error/Evaluation Error')
plt.title('Mini Project1:1.2 Ridge Regression-Errors vs lambda')
plt.legend()

plt.figure(2)
plt.scatter(numpy.squeeze(numpy.asarray(x1)), numpy.squeeze(numpy.asarray(y)), label='Given Data(Y)')
plt.plot(new_x1, new_y_prime, linestyle='--', color='red', label='Prediction(Y_Prime)')
plt.xlabel('X')
plt.ylabel('Y/Y_Prime')
plt.title('Mini Project1:1.2 Ridge Regression-Polynomial Fit')
plt.legend()

plt.show()

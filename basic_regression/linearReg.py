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


# Load Data
data = numpy.load('linRegData.npy')
data = numpy.matrix(data)

# Get X and Y
m = data.shape[0]
x1 = data[:, 0]
x = numpy.append(numpy.ones((m, 1),  dtype=x1.dtype), x1, axis=1)
y = data[:, 1]

# Calculate (X^T*X)^-1
temp1 = x.transpose().dot(x).I

# Calculate (X^T*Y)
temp2 = x.transpose().dot(y)

# Calculate  theta = ((X^T*X)^-1)*((X^T*Y))
theta = temp1.dot(temp2)

y_prime = x.dot(theta)

error = (numpy.power((y-y_prime), 2).sum())/m
print "Mean square error: " + str(error)

plt.plot(numpy.squeeze(numpy.asarray(x1)), numpy.squeeze(numpy.asarray(y)), label='Given Data(Y)')
plt.plot(numpy.squeeze(numpy.asarray(x1)), numpy.squeeze(numpy.asarray(y_prime)), linestyle='--', label='Prediction(Y_Prime)')
plt.xlabel('X')
plt.ylabel('Y/Y_Prime')
plt.title('Mini Project1:1.1 Linear Regression')
plt.legend()
plt.show()

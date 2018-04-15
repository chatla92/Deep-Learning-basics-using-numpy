
conv_net.py contains the network definition where as helper.py contains modules required for convolutional layer and linear layers


For Running a simple CNN:

* Set pool and dropout variables to False.

In Conv_net.py:
	line 35: pool = False
	line 39: dropout = False

For Running Maxpooling:

* Set pool variable to True and dropout variable to False.

In Conv_net.py:
	line 35: pool = True
	line 39: dropout = False


For Running Dropout:

* Set pool and dropout variables to True.

In Conv_net.py:
	line 35: pool = True
	line 39: dropout = True


import numpy as np
import os
import pdb
import skimage
from skimage.transform import resize as imresize

datasets_dir = '../data/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=False,subset=True,digit_range=[0,2],shuffle=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(float)

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	tempX = np.zeros((ntrain,10,10))
	for i in range(ntrain):
		tempX[i,:,:] = imresize(trX[i,:,:], (10,10), mode="constant")
	trX = np.reshape(tempX, (ntrain,10,10,1))

	tempX = np.zeros((ntest,10,10))
	for i in range(ntest):
		tempX[i,:,:] = imresize(teX[i,:,:], (10,10), mode="constant")
	teX = np.reshape(tempX, (ntest,10,10,1))

	trX = trX/255.
	teX = teX/255.

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)


	if subset:
		subset_label = np.arange(digit_range[0], digit_range[1])
		train_data_sub = []
		train_label_sub = []
		test_data_sub = []
		test_label_sub = []
		for i in subset_label:
			train_sub_idx = np.where(trY==i)
			test_sub_idx = np.where(teY==i)
			#pdb.set_trace()
			A = trX[train_sub_idx[0],:,:,:]
			C = teX[test_sub_idx[0],:,:,:]
			if onehot:
				B = trY[train_sub_idx[0],:]
				D = teY[test_sub_idx[0],:]
			else:
				B = trY[train_sub_idx[0]]
				D = teY[test_sub_idx[0]]
			
			train_data_sub.append(A)
			train_label_sub.append(B)
			test_data_sub.append(C)
			test_label_sub.append(D)

		trX = train_data_sub[0]
		trY = train_label_sub[0]
		teX = test_data_sub[0]
		teY = test_label_sub[0]
		for i in range(digit_range[1]-digit_range[0]-1):
			trX = np.concatenate((trX,train_data_sub[i+1]),axis=0)
			trY = np.concatenate((trY,train_label_sub[i+1]),axis=0)
			teX = np.concatenate((teX,test_data_sub[i+1]),axis=0)
			teY = np.concatenate((teY,test_label_sub[i+1]),axis=0)

		if shuffle:
			train_idx = np.random.permutation(trX.shape[0])
			test_idx = np.random.permutation(teX.shape[0])
			trX = trX[train_idx,:,:,:]
			teX = teX[test_idx,:,:,:]
			if onehot:
				trY = trY[train_idx,:]
				teY = teY[test_idx,:]
			else:
				trY = trY[train_idx]
				teY = teY[test_idx]

	trY = trY.reshape(1,-1)
	teY = teY.reshape(1,-1)
	trY = trY-digit_range[0]
	teY = teY-digit_range[0]
	return trX, trY, teX, teY		
'''
def main():
	trx, trY, teX, teY = mnist()

if __name__ == "__main__":
    main()
'''
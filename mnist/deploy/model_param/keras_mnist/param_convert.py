

import numpy as np
import h5py

f = h5py.File('mnist_model.h5', 'r')

# dset = f['/model_weights/conv2d/conv2d/bias:0']
dset = f['/model_weights/dense_1/dense_1/kernel:0']
print(dset)
print(dset.name)
data = np.array(dset)
print(data.shape)
print(data)
# data=data.swapaxes(1,3)
# data=data.swapaxes(0,2)
data=data.swapaxes(0,1)
# data=np.squeeze(data)
print(data.shape)
print(data)
# data=data.flatten()
# print('--------------------------------------------------')
# print(data.shape)
# print(data)
# print('--------------------------------------------------')
contiguous_array = np.ascontiguousarray(data)
print(contiguous_array.shape)
print(contiguous_array)
with open('dense_1_kerne.bin', 'wb') as file:
    file.write(contiguous_array)

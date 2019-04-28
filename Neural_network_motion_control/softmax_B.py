import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

mnist_strain = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

print(len(mnist_strain))
print(len(mnist_test))

fearture, label = mnist_strain[0:9]

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
num_workers = 4

train_iter = gdata.DataLoader(mnist_strain.transform_first(transformer),batch_size, shuffle=False, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for x,y in train_iter:
    continue
print("%.2f sec" %(time.time()-start))

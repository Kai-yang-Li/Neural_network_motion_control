import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
tranin_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

w.attach_grad()
b.attach_grad()

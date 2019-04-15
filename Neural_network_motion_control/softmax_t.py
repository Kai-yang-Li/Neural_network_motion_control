import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
tranin_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = (28*28)
num_outputs = 10

w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

w.attach_grad()
b.attach_grad()

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(axis =1, keepdims = True)
    return x_exp/partition

def net(x):
    return softmax( nd.dot(x.reshape((-1, num_inputs)),w)+b )

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()




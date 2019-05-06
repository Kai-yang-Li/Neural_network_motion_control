import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

##W1 = nd.random.normal(scale=0.01, shape=(784, 512))
##b1 = nd.zeros(512)
##W2 = nd.random.normal(scale=0.01, shape=(512, 256))
##b2 = nd.zeros(256)
##W3 = nd.random.normal(scale=0.01, shape=(256, 10))
##b3 = nd.zeros(10)
##params = [W1, b1, W2, b2, W3, b3]

W1 = nd.random.normal(scale=0.01, shape=(784, 256))
b1 = nd.zeros(256)
W2 = nd.random.normal(scale=0.01, shape=(256, 10))
b2 = nd.zeros(10)
params = [W1, b1, W2, b2]


for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)
    
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 50, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)






import d2lzh as d2l
from mxnet import autograd, gluon, init, nd 
from mxnet.gluon import loss as gloss, nn

def dropout(X, drop_prob): 
    assert 0 <= drop_prob <= 1 
    keep_prob = 1 - drop_prob 
    # 这种情况下把全部元素都丢弃 
    if keep_prob == 0:
        return X.zeros_like()
    mask = (nd.random.uniform(0, 1, X.shape) < keep_prob)
    return mask * X / keep_prob

'''
X = nd.arange(16).reshape((2, 8))
print(X)
print(dropout(X,0.2))
print(dropout(X,0.6))
'''

W1 = nd.random.normal(scale=0.01, shape=(784, 256))
b1 = nd.zeros(256)
W2 = nd.random.normal(scale=0.01, shape=(256, 256))
b2 = nd.zeros(256)
W3 = nd.random.normal(scale=0.01, shape=(256, 10))
b3 = nd.zeros(10)
params = [W1, W2, W3, b1, b2, b3]
for param in params:
    param.attach_grad()

drop_prob1, drop_prob2 = 0.1, 0.4

def net(x):
    x = x.reshape((-1, 784))
    H1 = ( nd.dot(x,W1)+b1 ).relu()
    if autograd.is_training():
        H1 = dropout(H1,drop_prob1)
    H2 = ( nd.dot(H1,W2)+b2 ).relu()
    if autograd.is_training():
        H2 = dropout(H2,drop_prob2)
    return nd.dot(H2, W3) + b3

num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

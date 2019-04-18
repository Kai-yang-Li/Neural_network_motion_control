import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

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
   return - nd.pick(y_hat, y).log()

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

print (d2l.evaluate_accuracy(test_iter, net))

num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs,batch_size,params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0 
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size) # “softmax回归的简洁实现”一节将用到
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
            test_acc = d2l.evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,[w, b], lr)

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy()) titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])


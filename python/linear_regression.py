
from mxnet import nd,autograd
import random

sample_nums = 1000
weight_nums =2
true_weights =nd.array([12,-3.4]).reshape(2,1)
true_bias=4.2
batch_size=5

def random_data_iter(features,labels,batch_size):
    indexs = list(range(len(features)))
    random.shuffle(indexs)
    for i in range(0,sample_nums,batch_size):
        random_sub_indexs = nd.array(indexs[i:min(i+batch_size,len(features))])
        yield features.take(random_sub_indexs),labels.take(random_sub_indexs)

features=nd.random.normal(shape=(sample_nums,weight_nums))

def linear_model(X,w,b):
    dot=nd.dot(X,w)
    labels=dot+b
    return labels

labels = linear_model(features,true_weights,true_bias)


def sgd(params,lr,batch_size):
    for param in params:
        param[:] = param-lr*param.grad/batch_size

def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2


epoch=5
lr=0.1
# train_bias=nd.zeros(shape=(1,))
train_bias=nd.random.normal(shape=(1,))
train_weights =nd.random.normal(shape=(weight_nums,1))
# train_weights = nd.ones(shape=(weight_nums,1))
print(train_weights)
print(train_bias)
train_bias.attach_grad()
train_weights.attach_grad()
for i in range(epoch):
    for X,y in random_data_iter(features,labels,batch_size):

        with autograd.record():
            loss_result = squared_loss(linear_model(X,train_weights,train_bias),y)
        loss_result.backward()
        sgd([train_weights,train_bias],lr,batch_size)
    whole_loss=squared_loss(linear_model(features,train_weights,train_bias),labels)
    print('<%d>loss is %f' %(i+1,whole_loss.mean().asnumpy()))
print(train_weights)
print(train_bias)

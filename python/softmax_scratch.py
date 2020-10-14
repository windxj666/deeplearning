import d2lzh
from mxnet import nd,autograd

batch_size=256
feature_nums=28*28
output_nums=10
train_iter, test_iter=d2lzh.load_data_fashion_mnist(batch_size)

def softmax(Y):
    Y_exp =Y.exp()
    exp_sum=Y_exp.sum(axis=1,keepdims=True)
    return Y_exp/exp_sum

def net(X,w,b):
    return softmax(nd.dot(X.reshape(-1,feature_nums),w)+b)

def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()

def sgd(params,lr,batch_size):
    for param in params:
        param[:]=param-lr*param.grad/batch_size

def eval_accuracy(y_hat,y):
    return (y_hat.argmax(axis=1)==y.astype('float32')).sum().asscalar()

epoch=5

lr=0.1
train_weights=nd.normal(scale=0.01,shape=(feature_nums,output_nums))
train_bias=nd.normal(scale=0.01,shape=(output_nums,))
train_weights.attach_grad()
train_bias.attach_grad()
for i in range(1,epoch+1):
    loss_total=0
    accuracy_total=0
    samples=0
    for X,y in train_iter:
        with autograd.record():
            loss_val = cross_entropy(net(X,train_weights,train_bias),y).sum()
        loss_val.backward()
        sgd([train_weights,train_bias],lr,batch_size)
        loss_total+=loss_val.asscalar()
        accuracy_total+=eval_accuracy(net(X,train_weights,train_bias),y)
        samples+=batch_size
        # print('loss_total:%.6f,accuracy_total:%.6f' %(loss_total,accuracy_total))
    print('epoch:%d,loss:%.6f,accurary:%.6f' %(i,loss_total/samples,accuracy_total/samples))
print(train_weights)
print(train_bias)
# for X, y in test_iter:
#     break
#
# true_labels = d2lzh.get_fashion_mnist_labels(y.asnumpy())
# pred_labels = d2lzh.get_fashion_mnist_labels(net(X,train_weights,train_bias).argmax(axis=1).asnumpy())
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#
# d2lzh.show_fashion_mnist(X[0:9], titles[0:9])

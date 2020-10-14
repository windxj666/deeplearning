from mxnet import gluon,init,autograd
from mxnet.gluon import loss as gloss,nn
import d2lzh

batch_size=256
train_iter,test_iter=d2lzh.load_data_fashion_mnist(batch_size)

net=nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.SoftmaxCrossEntropyLoss()
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
epoch=5
for i in range(epoch):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X,y in train_iter:
        with autograd.record():
            l=loss(net(X),y).sum()
        l.backward()
        trainer.step(batch_size)
        train_l_sum+=l.asscalar()
        n+=batch_size
    print('epoch %d,loss is %.6f' %(i+1,train_l_sum/n))



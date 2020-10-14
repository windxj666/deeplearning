import d2lzh
from mxnet import gluon,init,nd
from mxnet.gluon import loss as gloss,nn

batch_size=256
train_iter,test_iter=d2lzh.load_data_fashion_mnist(batch_size)
net=nn.Sequential()
net.add(nn.Dense(256,activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
loss =gloss.SoftmaxCrossEntropyLoss()
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})
num_epoch=5
d2lzh.train_ch3(net,train_iter,test_iter,loss,num_epoch,batch_size,None,None,trainer)

nd.concat(X,nd.power)
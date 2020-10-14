import mxnet
from mxnet import gluon,nd,init,autograd
from mxnet.gluon import nn,data as gdata,loss as gloss
# from mxnet.gluon import

sample_nums =1000
weight_nums=2
true_w = nd.array([4.5,-7.8])
true_b=8.9
batch_size=5
features = nd.random.normal(shape=(sample_nums,weight_nums))

def linear_model(X,w,b):
    return nd.dot(X,w)+b

labels =linear_model(features,true_w,true_b)

samples=gdata.ArrayDataset(features,labels)
data_batch=gdata.DataLoader(samples,batch_size,shuffle=True)
net =nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=1))
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})
loss=gloss.L2Loss()
epoch=3
for i in range(epoch):
    for X,y in data_batch:
        with autograd.record():
            l=loss(net(X),y)
        l.backward()
        trainer.step(batch_size)
    total_l=loss(net(features),labels)
    print('round %d loss is %f' %(i+1, total_l.mean().asnumpy()))

print(net[0].weight.data())
print(net[0].bias.data())
print(net[0].weight.grad())
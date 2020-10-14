from mxnet import gluon,autograd,init
from mxnet.gluon import data as gdata,loss as gloss,nn
import shutil
import os,d2lzh as d2l,time
import pandas as pd

batch_size=128
input_dir = 'D:\d2l-zh-0925\data\kaggle_cifar10\\train_valid_test'

transform_train=gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(40),
    gdata.vision.transforms.RandomResizedCrop(32,(0,64,1),(1.0,1.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
])

transform_test=gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
])

train_ds = gdata.vision.ImageFolderDataset(os.path.join(input_dir,'train'),flag=1)
valid_ds=gdata.vision.ImageFolderDataset(os.path.join(input_dir,'valid'),flag=1)
test_ds=gdata.vision.ImageFolderDataset(os.path.join(input_dir,'test'),flag=1)
train_valid_ds=gdata.vision.ImageFolderDataset(os.path.join(input_dir,'train_valid'),flag=1)

train_iter=gdata.DataLoader(train_ds.transform_first(transform_train),batch_size,shuffle=True)
valid_iter=gdata.DataLoader(valid_ds.transform_first(transform_test),batch_size,True)
train_valid_iter=gdata.DataLoader(train_valid_ds.transform_first(transform_train),batch_size,True)
test_iter=gdata.DataLoader(test_ds.transform_first(transform_test),batch_size,False)

class Residual(nn.HybridBlock):
    def __init__(self,num_channels,use_1x1conv=False,strides=1,**kwargs):
        super(Residual,self).__init__(**kwargs)
        self.conv1=nn.Conv2D(num_channels,kernel_size=3,strides=strides,padding=1)
        self.conv2=nn.Conv2D(num_channels,kernel_size=3,strides=1,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2D(num_channels,kernel_size=1,strides=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm()
        self.bn2=nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        return F.relu(Y + X)

# class Residual(nn.HybridBlock):
#     def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
#         super(Residual, self).__init__(**kwargs)
#         self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
#                                strides=strides)
#         self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
#                                    strides=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm()
#         self.bn2 = nn.BatchNorm()
#
#     def hybrid_forward(self, F, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         return F.relu(Y + X)

def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net

# def resnet18(num_classes):
#     net=nn.HybridSequential()
#     net.add(nn.Conv2D(channels=64,kernel_size=3,strides=1,padding=1),
#             nn.BatchNorm(),nn.Activation('relu'))
#
#     def res_block(num_channels,num_residuals,first_block=False):
#         blk=nn.HybridSequential()
#         for i in range(num_residuals):
#             if i==0 and not first_block:
#                 blk.add(Residual(num_channels,True,strides=2))
#             else:
#                 blk.add(Residual(num_channels))
#         return blk
#
#     net.add(res_block(64,2,True),
#             res_block(128,2),
#             res_block(256,2),
#             res_block(512,2))
#     net.add(nn.GlobalAvgPool2D(),
#             nn.Dense(num_classes))
#     return net

loss = gloss.SoftmaxCrossEntropyLoss()
ctx=d2l.try_gpu()

def train(net,train_iter,valid_iter,lr,wd,lr_decay,lr_period,epochs,loss,ctx):
    trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr,'wd':wd,'momentum':0.9})
    for epoch in range(epochs):
        l_sum,acc_sum,n,start=.0,.0,0,time.time()
        if epoch>0 and epoch%lr_period==0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        for X,y in train_iter:
            y=y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat=net(X.as_in_context(ctx))
                l=loss(y_hat,y).sum()
            l.backward()
            trainer.step(batch_size)
            l_sum+=l
            acc_sum+=(y_hat.argmax(axis=1)==y).sum().asscalar()
            n+=y.size
        train_time = time.time()
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy(valid_iter, net, ctx)
            print("epoch %d,valid_acc %f" %(epoch,valid_acc))
        print("epoch %d,cost %f sec,loss %f,train_acc %f,lr %f" %(epoch,train_time,l_sum/n,acc_sum/n,trainer.learning_rate))

lr,wd,lr_decay,lr_period,epochs=0.1,5e-4,.1,80,50
num_classes=10
# net=resnet18(num_classes)
# net.initialize(init=init.Xavier(),ctx=ctx)
# net.hybridize()
# train(net,train_iter,valid_iter,lr,wd,lr_decay,lr_period,epochs,loss,ctx)
# after train chose appropriate model
net=resnet18(num_classes)
net.initialize(ctx=ctx,init=init.Xavier())
net.hybridize()
train(net,train_valid_iter,None,lr,wd,lr_decay,lr_period,epochs,loss,ctx)
labels=[]
for X,_ in test_iter:
    y_hat=net(X.as_in_context(ctx))
    labels.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sort_ids=list(range(1,len(test_ds)+1))
sort_ids=sort_ids.sort(key=lambda x:str(x))
df=pd.DataFrame({'ids':sort_ids,'labels':labels})
df['labels'].apply(lambda x:train_valid_ds.synsets[x])
df.to_csv('submission.csv',index=False)


















































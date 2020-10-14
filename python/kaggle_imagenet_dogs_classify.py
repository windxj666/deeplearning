import collections
import os,time,d2lzh as d2l
from mxnet import nd,init,autograd,gluon
from mxnet.gluon import model_zoo,nn,loss as gloss

def get_net(ctx):
    finetune_net=model_zoo.vision.resnet34_v2(pretrained=True)
    finetune_net.output_new=nn.HybridSequential()
    finetune_net.output_new.add(nn.Dense(256,activation='relu'))
    finetune_net.output_new.add(nn.Dense(120))
    finetune_net.output_new.initialize(init=init.Xavier(),ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net
def evaluate_loss(net,valid_iter,ctx,loss):
    l,n=.0,0
    for X,y in valid_iter:
        y=y.as_in_context()
        features_output= net.features(X.as_in_context(ctx))
        output=net.output_new(features_output)
        l+=loss(output,y).sum()
        n+=y.size()
    return l/n
def train(net,train_iter,valid_iter,loss,epoches,lr,lr_decay,lr_period,wd,ctx,batch_size):
    trainer=gluon.Trainer(net.output_new.collect_params(),'sgd',{'learning_rate':lr,'wd':wd})
    for epoch in range(epoches):
        if epoch>0 and epoch%lr_period==0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        l_sum,n,start=.0,0,time.time()
        for X,y in train_iter:
            y=y.as_in_context(ctx)
            features_output=net.features(X.as_in_context(ctx))
            with autograd.record():
                output=net.output_new(features_output)
                l=loss(output,y).sum()
            l.backward()
            trainer.step(batch_size)
            l_sum+=loss(output,y).sum()
            n+=y.size()
        cost_str="time %f sec" %(time.time()-start)
        if valid_iter is not None:
            valid_loss=evaluate_loss(net,valid_iter,ctx,loss)
            print_str=("epoch %d,train_loss %f,valid_loss %f," % (epoch,l_sum/n,valid_loss))
        else:
            print_str=("epoch %d,train_loss %f," %(epoch,l_sum/n))
        print(print_str+cost_str+',lr '+str(trainer.learning_rate))
ctx=d2l.try_gpu()
net=get_net(ctx)
loss= gloss.SoftmaxCrossEntropyLoss()
epoches=10
#todo
#variable init
train(net,train_iter,None,loss,epoches,lr,lr_decay,lr_period,wd,ctx,batch_size)

preds=[]
for data,_ in test_iter:
    output_features=net.features(data.as_in_context(ctx))
    output=net.output_new(output_features)
    output=nd.softmax(output)





net = get_net(ctx)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_context(ctx))
    output = nd.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('submission-dog.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
import d2lzh
import mxnet
from mxnet import nd
from mxnet import autograd
import pandas as pd

# for i in range(3):
#     print(i)
#
# exit()
train_data = pd.read_csv('D:\d2l-zh-0925\data\kaggle_house_pred_train.csv')
test_data = pd.read_csv('D:\d2l-zh-0925\data\kaggle_house_pred_test.csv')
all_feature = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_feature.shape)

numerial_index = all_feature.dtypes[all_feature.dtypes != 'object'].index
# standardization numerial data
all_feature[numerial_index] = all_feature[numerial_index].apply(lambda X: (X - X.mean()) / X.std())
all_feature[numerial_index] = all_feature[numerial_index].fillna(0)
all_feature=pd.get_dummies(all_feature,dummy_na=True)
train_nums=train_data.shape[0]
train_features=mxnet.nd.array(all_feature[:train_nums].values)
valid_features=mxnet.nd.array(all_feature[train_nums:].values)
train_labels=mxnet.nd.array(train_data.iloc[:,-1].values).reshape(-1,1)

def log_sqrt(y_hat,y):
    y_hat=nd.clip(y_hat,1,float('inf'))
    return nd.sqrt(((nd.log(y_hat)-nd.log(y))**2).mean()).asscalar()

def get_k_fold_data(k,i,X,y):
    assert k>1
    train_X,train_y=None,None
    per_size = X.shape[0]//k
    for j in range(k):
        idx=slice(j*per_size,(j+1)*per_size)
        if i==j:
            valid_X=X[idx]
            valid_y=y[idx]
        elif train_X is None:
            train_X=X[idx]
            train_y=y[idx]
        else:
            train_X=mxnet.nd.concat(train_X,X[idx],dim=0)
            train_y=mxnet.nd.concat(train_y,y[idx],dim=0)
    return train_X,train_y,valid_X,valid_y

def train_iter(net,train_features,train_labels,test_features,test_labels,epochs,lr,wd,batch_size):
    train_loss_arr,test_loss_arr=[],[]

    trainer_w = mxnet.gluon.Trainer(net.collect_params('.*weight'), 'adam', {'learning_rate': lr, 'wd': wd})
    trainer_b = mxnet.gluon.Trainer(net.collect_params('.*bias'), 'adam', {'learning_rate': lr})
    for epoch in range(epochs):
        dataset = mxnet.gluon.data.ArrayDataset(train_features, train_labels)
        data_iter = mxnet.gluon.data.DataLoader(dataset, batch_size, shuffle=True)
        for X,y in data_iter:
            with mxnet.autograd.record():
                train_loss=loss(net(X),y)
            train_loss.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        # train_loss_epoch=loss(net(train_features), train_labels).mean().asscalar()
        train_loss_epoch = loss_valid(net(train_features), train_labels)
        train_loss_arr.append(train_loss_epoch)
        print('epoch:%d,train_loss:%f' %(epoch,train_loss_epoch))
        if test_labels is not None:
            test_loss_epoch=loss_valid(net(test_features),test_labels)
            test_loss_arr.append(test_loss_epoch)
            print('test_loss:%f' %(test_loss_epoch))
    return train_loss_arr,test_loss_arr

def getNet():
    net = mxnet.gluon.nn.Sequential()
    net.add(mxnet.gluon.nn.Dense(1))
    net.initialize()
    return net

epochs=300
lr=5
wd=.0
batch_size=64
k=6

loss=mxnet.gluon.loss.L2Loss()
loss_valid=log_sqrt

train_loss_sum, test_loss_sum = 0.0, 0.0
for fold in range(k):
    train_features_k,train_labels_k,test_features_k,test_labels_k=get_k_fold_data(k,fold,train_features,train_labels)
    net=getNet()
    train_loss_arr,test_loss_arr=train_iter(net,train_features_k,train_labels_k,test_features_k,test_labels_k,epochs,lr,wd,batch_size)
    print('k-fold:%d,train_loss_last:%f' %(fold,train_loss_arr[-1]))
    print('k-fold:%d,test_loss_last:%f' % (fold, test_loss_arr[-1]))
    train_loss_sum+=train_loss_arr[-1]
    test_loss_sum+=test_loss_arr[-1]
print('k-fold avg train loss:%f,test loss:%f' %(train_loss_sum/k,test_loss_sum/k))

def train_and_pred():
    net =getNet()
    train_iter(net,train_features,train_labels,None,None,epochs,lr,wd,batch_size)
    valid_labels=net(valid_features).asnumpy()
    test_data['SalePrice']=pd.Series(valid_labels.reshape(1,-1)[0])
    valid_labels=pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    valid_labels.to_csv('kaggle_house_price_pred',index=False)

train_and_pred()






















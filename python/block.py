import mxnet
from mxnet import ndarray as nd
from mxnet.gluon import nn

class MLP(nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden=nn.Dense(256,activation='relu')
        self.output=nn.Dense(10)

    def forward(self,X):
        return self.output(self.hidden(X))
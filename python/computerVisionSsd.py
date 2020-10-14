import mxnet as mx
from mxnet import gluon, nd, init, image, autograd,contrib
from mxnet.gluon import nn, loss as gloss, data as gdata
import d2lzh as d2l
import time
import matplotlib.pyplot as plot

def half_W_H(num_channels):
    block = nn.Sequential()
    block.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
              nn.BatchNorm(in_channels=num_channels),
              nn.Activation('relu'))
    block.add(nn.MaxPool2D(2))
    return block


def base_net():
    block = nn.Sequential()
    for channels in [32, 64, 128]:
        block.add(half_W_H(channels))
    return block


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = half_W_H(128)
    return blk

def class_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


def transpose_flatten(X):
    return X.transpose((0, 2, 3, 1)).flatten()


def concat_preds(Xs):
    return nd.concat(*[transpose_flatten(X) for X in Xs], dim=1)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


def blk_forward(X, blk, sizes, ratios, class_predictor, bbox_predictor):
    Y = blk(X)
    anchors = contrib.ndarray.MultiBoxPrior(Y, sizes=sizes, ratios=ratios)
    class_preds = class_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, class_preds, bbox_preds)


class TinySsd(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySsd, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'class_predictor_%d' % i, class_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_predictor_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, class_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], class_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, 'blk_%d' % i), sizes[i],
                                                                       ratios[i],
                                                                       getattr(self, 'class_predictor_%d' % i),
                                                                       getattr(self, 'bbox_predictor_%d' % i))
        return (nd.concat(*anchors, dim=1), concat_preds(class_preds).reshape(0, -1, self.num_classes + 1),
                concat_preds(bbox_preds))


batch_size = 32
img_size = 256
train_iter = image.ImageDetIter(path_imgrec='D:\d2l-zh20200904\data\pikachu\\train.rec',
                                path_imgidx='D:\d2l-zh20200904\data\pikachu\\train.idx',
                                batch_size=batch_size, data_shape=(3, img_size, img_size), shuffle=True, rand_crop=1,
                                min_object_covered=0.95, max_attempts=200)
# test_iter = image.ImageDetIter(path_imgrec='D:\d2l-zh20200904\data\pikachu\\val.rec', batch_size=batch_size,
#                                data_shape=(3, img_size, img_size), shuffle=False)

softmaxloss = gloss.SoftmaxCrossEntropyLoss()
l1loss = gloss.L1Loss()
def ssdLoss(class_preds, bbox_preds, class_labels, bbox_labels, bbox_mask):
    class_loss = softmaxloss(class_preds, class_labels)
    bbox_loss = l1loss(bbox_preds * bbox_mask, bbox_labels * bbox_mask)
    return class_loss + bbox_loss

epoches = 20
net = TinySsd(1)

ctx = d2l.try_gpu()
net.initialize(init.Xavier(),ctx=ctx)
lr = .2
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr,'wd': 5e-4})

for epoch in range(epoches):
    class_acc, bbox_acc, n, m = 0, 0, 0, 0
    start=time.time()
    train_iter.reset()
    for batch in train_iter:
        X = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            anchors, class_preds, bbox_preds = net(X)
            bbox_labels, bbox_mask, class_labels = contrib.nd.MultiBoxTarget(anchors, y,
                                                                             class_preds.transpose((0, 2, 1)))
            l = ssdLoss(class_preds, bbox_preds, class_labels, bbox_labels,bbox_mask)
        l.backward()
        trainer.step(batch_size)
        class_acc += (class_preds.argmax(axis=-1)==class_labels).sum().asscalar()
        n+=class_labels.size
        bbox_acc +=((bbox_preds-bbox_labels)*bbox_mask).abs().sum().asscalar()
        m+=bbox_labels.size
    print('epoch %d,class acc is %.2e,bbox err is %.2e,time is %.2f' % (epoch+1,class_acc/n,bbox_acc/m,time.time()-start))
img=image.imread('D:\d2l-zh20200904\img\pikachu.jpg')
feature=image.imresize(img,256,256).astype('float32').transpose((2,0,1)).expand_dims(axis=0)
anchors,class_preds,bbox_preds=net(feature)
class_probs=class_preds.softmax(axis=-1).transpose((0,2,1))
output=contrib.nd.MultiBoxDetection(class_probs,bbox_preds,anchors)
idx=[i for i,row in enumerate(output[0]) if row[0].asscalar() != -1]
fig=plot.imshow(img.asnumpy())
nms_outputs=output[0,idx]
h,w=img.shape[0:2]
for nms_output in nms_outputs:
    if nms_output[1]<0.3:
        continue
    d2l.show_bboxes(fig.axes, [nms_output[2:]*nd.array((w,h,w,h))],'%.2f' % nms_output[1].asscalar())
plot.show()

exit()

import d2lzh as d2l
import sys
import mxnet as mx
from mxnet import gluon, nd, init, autograd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils

aug_method = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomFlipLeftRight(),
                                              gdata.vision.transforms.RandomFlipTopBottom(),
                                              gdata.vision.transforms.RandomHue(),
                                              gdata.vision.transforms.RandomContrast(),
                                              gdata.vision.transforms.RandomBrightness(),
                                              gdata.vision.transforms.RandomSaturation(),
                                              gdata.vision.transforms.ToTensor()])
no_aug = gdata.vision.transforms.Compose(gdata.vision.transforms.ToTensor())

num_workers = 0 if sys.platform.startswith('win32') else 4


def getCifar10(is_train, batch_size, augs):
    return gdata.DataLoader(gdata.vision.CIFAR10(train=is_train).transform_first(augs), batch_size=batch_size,
                            shuffle=is_train, num_workers=num_workers)


def try_all_gpus():
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx),
            features.shape[0])


def evaluate_accuracy(net, data_iter, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = 0, 0
    for batch in data_iter:
        Xs, ys, _ = _get_batch(batch, ctx)
        for X, y in zip(Xs, ys):
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


net = d2l.resnet18(10)
loss = gloss.SoftmaxCrossEntropyLoss()
ctx = try_all_gpus()
net.initialize(ctx=ctx, init=init.Xavier())
batch_size = 1000
train_iter = getCifar10(True, batch_size, aug_method)
test_iter = getCifar10(False, batch_size, no_aug)
lr = 0.01
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})


def train(net, train_iter, test_iter, num_epoches, ctxes, loss, trainer):
    train_acc_sum, train_l_sum,acc_count,l_count = 0, 0, 0,0
    for epoch in range(num_epoches):
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctxes)
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
            acc_count+=sum([y_hat.size for y_hat in y_hats])
            train_l_sum+=sum([l.sum().asscalar() for l in ls])
            l_count+=sum([l.size for l in ls])

        test_acc=evaluate_accuracy(net,test_iter,ctx)
        print('.......')

from mxnet.gluon import model_zoo,data as gdata,loss as gloss
from mxnet import init,gluon
import d2lzh as d2l

train_data =gdata.vision.ImageFolderDataset('...')
test_data = gdata.vision.ImageFolderDataset('...')

pretrained_model=model_zoo.vision.resnet18_v2(pretrained=True)
finetuning_model=model_zoo.vision.resnet18_v2(classes=2)
finetuning_model.features=pretrained_model.features
finetuning_model.output.initialize(init.Xavier())
finetuning_model.output.collect_params().setattr('lr_mult',10)

normalize=gdata.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs=gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomResizedCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize
])
test_augs=gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    normalize
])
batch_size=256
train_iter=gdata.DataLoader(train_data.transform_first(train_augs),batch_size=batch_size,shuffle=True)
test_iter=gdata.DataLoader(test_data.transform_first(test_augs),batch_size=batch_size)
loss = gloss.SoftmaxCrossEntropyLoss()
ctx=d2l.try_all_gpus()
finetuning_model.collect_params().reset_ctx(ctx)
finetuning_model.hybridize()
trainer=gluon.Trainer(finetuning_model.collect_params(),'sgd',{'learning_rate':0.01,'wd':0.001})
num_epochs=5
d2l.train(train_iter,test_iter,finetuning_model,loss,trainer,ctx,num_epochs)
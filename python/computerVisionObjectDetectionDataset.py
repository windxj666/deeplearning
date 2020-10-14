import d2lzh as d2l
from mxnet import image
from matplotlib import pyplot as plt

batch_size=32
img_size=512
train_iter=image.ImageDetIter(path_imgrec='D:\d2l-zh20200904\data\pikachu\\train.rec',
                              path_imgidx='D:\d2l-zh20200904\data\pikachu\\train.idx',
                              batch_size=batch_size,data_shape=(3,img_size,img_size),shuffle=True,rand_crop=1,
                   min_object_covered=0.95,max_attempts=200)
test_iter=image.ImageDetIter(path_imgrec='D:\d2l-zh20200904\data\pikachu\\val.rec',batch_size=batch_size,data_shape=(3,img_size,img_size),shuffle=False)
batch=train_iter.next()
imgs=batch.data[0].transpose((0,2,3,1))/255
axes=d2l.show_images(imgs,4,8).flatten()

# imgs = (batch.data[0][0:10].transpose((0, 2, 3, 1))) / 255
# axes = d2l.show_images(imgs, 2, 5).flatten()
# for ax, label in zip(axes, batch.label[0][0:10]):
#     d2l.show_bboxes(ax, [label[0][1:5] * 256], colors=['w'])


labels=batch.label[0]
for ax,label in zip(axes,labels):
    d2l.show_bboxes(ax,[label[0][1:5]*img_size])
plt.show()
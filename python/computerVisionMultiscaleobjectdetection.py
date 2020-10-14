import d2lzh as d2l
from mxnet import contrib, image, nd
import matplotlib.pyplot as plt

img = image.imread('D:\d2l-zh-0925\img\catdog.jpg')
h, w = img.shape[0:2]
print(h,w)
d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_h, fmap_w))  # 前两维的取值不影响输出结果
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    print('anchor',anchors)
    bbox_scale = nd.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
display_anchors(fmap_w=2,fmap_h=3, s=[0.25])
plt.show()
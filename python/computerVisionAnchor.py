from mxnet import image,gluon,nd,autograd,contrib
import d2lzh as d2l

img=image.imread('D:\d2l-zh-0925\img\catdog.jpg').asnumpy()
h,w=img.shape[0:2]
img_random = nd.random.uniform(shape=(1,3,h,w))
anchors=contrib.ndarray.MultiBoxPrior(img_random,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5])
boxes=anchors.reshape(shape=(h,w,5,4))
box_scale=nd.array((w,h,w,h))
fig=d2l.plt.imshow(img)
d2l.show_bboxes(fig.axes,boxes[123,123,:,:]*box_scale)
ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
labels=contrib.nd.MultiBoxTarget(anchors,ground_truth.expand_dims(axis=0),nd.zeros((1,3,2042040)))


anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
output = contrib.ndarray.MultiBoxDetection(
    cls_probs.expand_dims(axis=0), offset_preds.expand_dims(axis=0),
    anchors.expand_dims(axis=0), nms_threshold=0.5)
output



import mxnet
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss
import d2lzh as d2l
import time,math

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

def to_onehot(X, char_size):
    output = []
    for x in X.T:
        output.append(nd.one_hot(x, char_size))
    return output


def getparams(vocab_size, hidden_nums):
    w_ih = nd.random.normal(scale=0.01, shape=(vocab_size, hidden_nums))
    w_hh = nd.random.normal(scale=0.01, shape=(hidden_nums, hidden_nums))
    w_ho = nd.random.normal(scale=0.01, shape=(hidden_nums, vocab_size))
    b_h = nd.zeros(hidden_nums, )
    b_o = nd.zeros(vocab_size, )
    params = [w_ih, w_hh, b_h, w_ho, b_o]
    for param in params:
        param.attach_grad()
    return params


def rnn(input, params, state):
    (w_ih, w_hh, b_h, w_ho, b_o) = params
    # print(params)
    # print(input)
    # exit()
    (H,)=state
    output = []
    for x in input:
        H = nd.tanh(nd.dot(x, w_ih) + nd.dot(H, w_hh) + b_h)
        Y = nd.dot(H, w_ho) + b_o
        output.append(Y)
    return output, (H,)


def rnn_predit(params, predict_step, char_prefix, vocab_size,num_hiddens):
    output = [char_to_idx[char_prefix[0]]]
    H = nd.zeros(shape=(1, num_hiddens))
    # params =getparams(vocab_size,hidden_nums)
    for i in range(predict_step + len(char_prefix) - 1):
        x = to_onehot(nd.array([output[-1]]), vocab_size)
        Y, (H,) = rnn(x, params, (H,))
        if i<len(char_prefix)-1:
            output.append(char_to_idx[char_prefix[i+1]])
        else:
            max_y=int(Y[0].argmax(axis=1).asscalar())
            output.append(max_y)
    print('predit string: ', ''.join([idx_to_char[index] for index in output]))


def gradient_clip(params, theta):
    norm = nd.array([0])
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = nd.sqrt(norm).asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_rnn(vocab_indices, is_random, vocab_size, hidden_nums, batch_size, num_epochs, num_steps, params,theta,lr,prefixs,predict_step):
    if is_random:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        l_sum=0.0
        n=0
        start = time.time()
        if not is_random:
            H = nd.zeros(shape=(batch_size, hidden_nums))
        data_iter = data_iter_fn(vocab_indices, batch_size, num_steps)
        for X, y in data_iter:
            if not is_random:
                for s in (H,):
                    s.detach()
            else:
                H = nd.zeros(shape=(batch_size, hidden_nums))
            with autograd.record():
                input = to_onehot(X, vocab_size)
                output, (H,) = rnn(input, params, (H,))
                output = nd.concat(*output, dim=0)
                y = y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            gradient_clip(params,theta)
            d2l.sgd(params,lr,1)
            l_sum+=l.asscalar()*y.size
            n+=y.size

        print('epoch %d,perplexity %s,time %s' %(epoch,math.exp(l_sum/n),time.time()-start))
        rnn_predit(params,predict_step,prefixs,vocab_size,hidden_nums)
d2l.RNNModel
is_random=True
hidden_nums=256
batch_size=32
num_epochs=250
num_steps=35
params=getparams(vocab_size,hidden_nums)
theta=1e-2
lr=1e2
prefixs='分开'
predict_step=50

train_rnn(corpus_indices,is_random,vocab_size,hidden_nums,batch_size,num_epochs,num_steps,params,theta,lr,prefixs,predict_step)

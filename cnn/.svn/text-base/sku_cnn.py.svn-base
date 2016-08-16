#coding=utf-8
import os
import sys
import time
import numpy
from itertools import *

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from io_module.sku_io import SkuTextReadStream
from io_module.model_io import load_pickle_model, dump_pickle_model

from lib.hidden_layer import HiddenLayer
from lib.convolutional_layer import LeNetConvPoolLayer
from lib.distance_layer import *
from lib.auc import auc

from cnn_conf import *
import json

def batch_auc(batch_sim_y):
    if len(batch_sim_y) == 0:
        return None
    #获取(sim, y)数组
    flat_sim_y = []
    for sims, ys in batch_sim_y:
        for sim, y in izip(sims, numpy.array(ys)):
            flat_sim_y.append((1, y, sim))

    return auc(flat_sim_y), numpy.mean([i[1] for i in flat_sim_y]), numpy.mean([i[2] for i in flat_sim_y])



def create_net(shared_xs, shared_y, sku_num, conf):
    #----------------------------与网络相关的配置------------------------------
    #相关符号变量
    index = T.lscalar()  # index to a [mini]batch
    learning_rate_var = T.fscalar()
    xs = tuple(T.ftensor4('x' + str(i)) for i in range(sku_num))
    y = T.fvector('y')  # the labels are presented as 1D vector of
    
    #----------------------------网络结构定义------------------------------
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (1 , 200-7+1) = (1, 194)
    # maxpooling reduces this further to (1, 194) = (1,97)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 1, 97)
    layer0 = LeNetConvPoolLayer(
        conf.rng,
        input=tuple(x.reshape(conf.sku_shape) for x in xs),
        image_shape=conf.layer0_image_shape,
        filter_shape=conf.layer0_filter_shape,
        poolsize=conf.layer0_poolsize,
        activation=conf.activation_func,
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (1,97-6+1) = (1, 92)
    # maxpooling reduces this further to (1, 92) = (1, 46)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 1, 46)
    layer1 = LeNetConvPoolLayer(
        conf.rng,
        input=layer0.output,
        image_shape=conf.layer1_image_shape,
        filter_shape=conf.layer1_filter_shape,
        poolsize=conf.layer1_poolsize,
        activation=conf.activation_func,
    )
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 1, 46)
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        conf.rng,
        input=tuple(o.flatten(2) for o in layer1.output),
        n_in=conf.layer2_n_in,
        n_out=conf.layer2_n_out,
        activation=conf.activation_func,
    )

    layer3 = HiddenLayer(
        conf.rng,
        input=layer2.output,
        n_in=conf.layer3_n_in,
        n_out=conf.layer3_n_out,
    )
    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    if sku_num == 2:
        layer4 = PairDistanceLayer(input=layer3.output, dis_type=conf.dis_type, cost_type=conf.cost_type)
    elif sku_num == 3:
        layer4 = TripleDistanceLayer(input=layer3.output, dis_type=conf.dis_type, cost_type=conf.cost_type)
    else:
        raise Exception('not support sku_num=%d' % sku_num)

    # the cost we minimize during training is the NLL of the model
    #cost = layer3.cost_prob(y)

    # create a list of all model parameters to be fit by gradient descent
    params =  layer3.params + layer2.params + layer1.params + layer0.params
    #delta_params
    dparams = []
    for p in params:
        v = numpy.zeros_like(p.get_value())
        dp = theano.shared(value=v, borrow=True)
        dparams.append(dp)

    # 损失函数
    cost = layer4.cost(y)

    #引入正则项
    for p in params:
        cost += conf.w_lambda * T.sum(p.flatten()**2)

    # 梯度
    grads = T.grad(cost, params)

    # updates
    import collections
    updates = collections.OrderedDict()
    for dparam, gparam in zip(dparams, grads):
        updates[dparam] = conf.momentum * dparam - gparam * learning_rate_var
    for dparam, param in zip(dparams, params):
        updates[param] = param + updates[dparam]

    givens = dict(zip(xs+(y,), shared_xs+(shared_y,)))
    train_model = theano.function(
            [index, learning_rate_var],
            (cost, layer4.cost(y)),
            updates=updates,
            givens=givens,
            on_unused_input='ignore'
    )

    test_model = theano.function(
        [index],
        (layer4.sim(y), y),
        givens=givens,
        on_unused_input='ignore'
    )

    vec_model = theano.function(
        [index],
        layer3.output,
        givens=givens,
        on_unused_input='ignore'
    )

    print "build model success"
    return train_model, test_model, vec_model, params

def create_shared_input(sku_num, conf):
    shared_y = theano.shared(numpy.zeros(conf.batch_size,dtype=theano.config.floatX), borrow=True)
    shared_xs = tuple(theano.shared(numpy.zeros(conf.sku_shape,dtype=theano.config.floatX), borrow=True) for i in range(sku_num))
    return shared_xs, shared_y
   
def evaluate_lenet5():
    #----------------------与训练相关的配置-----------------------
    #输入文件
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    sku_info_file = sys.argv[4]
    sku_num = int(sys.argv[5])
    info_tag = sys.argv[6]

    #还原模型和配置
    if len(sys.argv) >= 8:
        sav_model_file = sys.argv[7]
        try:
            params_value, conf = load_pickle_model(sav_model_file)
            print >> sys.stderr, '成功载入上次的参数和配置'
            #强制更新部分配置
            conf.reload()
            print >> sys.stderr, '-' * 10, 'conf', '-' * 10
            for k, v in conf.__dict__.iteritems():
                print >> sys.stderr, k, '=', v
            print >> sys.stderr, '-' * 20
        except:
            params_value, = load_pickle_model(sav_model_file)
            conf = Conf(info_tag)
            print >> sys.stderr, '成功载入上次的参数'
    else:
        params_value = None
        conf = Conf(info_tag)
        print >> sys.stderr, '重新训练模型'

    #学习率初始化
    learning_rate = conf.learning_rate_ini

    #输入层
    #----------------------模型生成-----------------------
    shared_xs, shared_y = create_shared_input(sku_num, conf)
    train_model, test_model, vec_model, params = create_net(shared_xs, shared_y, sku_num, conf)
    
    #还原模型参数
    if params_value:
        for p, v in izip(params, params_value):
            p.set_value(v)

    #----------------------测试模型-----------------------
    def run_test_model():
        #预估训练数据
        test_stream = SkuTextReadStream(test_file, sku_info_file, conf.sku_shape, conf.info_tag)
        batch_sim_y = []
        while not test_stream.is_finish:
            _y, _xs, _sku_ids = test_stream.read_data(sku_num)
            shared_y.set_value(_y.get_value())
            for shared_x, _x in izip(shared_xs, _xs):
                shared_x.set_value(_x.get_value())
                
            if shared_y.get_value().shape[0] != conf.batch_size:
                break

            batch_sim_y.append(test_model(0))

        #计算auc
        print 'final_auc=', batch_auc(batch_sim_y)
  
    #----------------------训练模型-----------------------
    num =0
    for i in range(conf.scan_num):
        print >>sys.stderr, 'scan_num', i
        train_stream = SkuTextReadStream(train_file, sku_info_file, conf.sku_shape, conf.info_tag)
        while not train_stream.is_finish:
            _y, _xs, _sku_ids = train_stream.read_data(sku_num) 
            shared_y.set_value(_y.get_value())
            for shared_x, _x in izip(shared_xs, _xs):
                shared_x.set_value(_x.get_value())
            
            if shared_y.get_value().shape[0] != conf.batch_size:
                break

            cost_ij, mse_ij = train_model(0, learning_rate)
            print 'training num=%d cost=%s mse=%s learning_rate=%f' % (num, cost_ij,mse_ij,learning_rate)

            num += 1
            if num % conf.dump_step == 0:
                conf.learning_rate_ini = learning_rate
                dump_pickle_model(model_file + '-' + str(num), [p.get_value() for p in params], conf)
            if num % conf.learning_decay_step == 0 and learning_rate > conf.min_learning_rate:
                learning_rate *= conf.learning_decay_fac

            if num % conf.eval_auc_step == 0:
                run_test_model() 
    
    #保存最终模型
    run_test_model()
    conf.learning_rate_ini = learning_rate
    dump_pickle_model(model_file, [p.get_value() for p in params], conf)

if __name__ == '__main__':
    evaluate_lenet5()

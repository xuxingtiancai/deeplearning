#coding=utf-8
import sys
sys.path.append('./')

import numpy
import theano
import theano.tensor as T
import logging
from itertools import *

from io_module.sku_io import SkuTextReadStream
from io_module.model_io import *

from cnn_conf import *
from sku_cnn import create_shared_input, create_net, batch_auc

def dump_vec():
    #输入参数
    sku_text_file = sys.argv[1]
    model_file = sys.argv[2]
    vec_file = sys.argv[3]
    sku_info_file = sys.argv[4]
    sku_num = int(sys.argv[5])
    info_tag = sys.argv[6]

    #还原模型和配置
    try:
        params_value, conf = load_pickle_model(model_file)
    except:
        params_value, = load_pickle_model(model_file)
        conf = Conf(info_tag)

    #生成模型
    shared_xs, shared_y = create_shared_input(sku_num, conf)
    train_model, test_model, vec_model, params = create_net(shared_xs, shared_y, sku_num, conf)

    #还原模型参数
    for p, v in izip(params, params_value):
        p.set_value(v)

    #读入初始数据
    test_stream = SkuTextReadStream(sku_text_file, sku_info_file, conf.sku_shape, conf.info_tag)
    fout = open(vec_file, 'w')
    batch_sim_y = []
    
    #生成所有向量
    sku = None
    while not test_stream.is_finish:
        _y, _xs, _sku_ids = test_stream.read_data(sku_num)
        shared_y.set_value(_y.get_value())
        for shared_x, _x in izip(shared_xs, _xs):
            shared_x.set_value(_x.get_value())   

        if shared_y.get_value().shape[0] != conf.batch_size:
            break

        #生成向量
        vecs = vec_model(0)
        for j in range(conf.batch_size):
            for i in range(sku_num):
                if _sku_ids[i][j] == sku:
                    continue
                sku = _sku_ids[i][j]
                vec = vecs[i][j]
                print >>fout, ' '.join(str(i) for i in [sku] + vec.tolist())

        #测试
        batch_sim_y.append(test_model(0))

    print 'final_auc=', batch_auc(batch_sim_y)

    fout.flush()
    fout.close()

if __name__ == '__main__':
    dump_vec()



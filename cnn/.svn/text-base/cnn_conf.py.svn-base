#coding=utf-8
import numpy
import theano
import theano.tensor as T

def relu(x):
    return T.switch(x<0, 0, x)
def leaky_relu(x):
    return T.switch(x<0, 0.3*x, x)

class Conf:
    def reload(self):
        #扫描次数
        self.dump_step = 500000 / self.batch_size
        self.eval_auc_step = 40000
        self.scan_num = 6

    #与网络结构相关的变量
    def __init__(self, info_tag):
        #batch
        self.batch_size = 64
        
        #学习相关参数
        self.learning_rate_ini = 0.1
        self.learning_decay_step = 300000 / self.batch_size
        self.learning_decay_fac = 0.98
        self.min_learning_rate = 0.000001
        self.activation_func = leaky_relu
        self.dis_type = 'cos' #'euler'
        self.cost_type = 'cost_mse' #'cost_prob'
        self.momentum = 0.5
        self.w_lambda = 0.01 / self.batch_size
        self.rng = numpy.random.RandomState(23455)

        #窗口尺寸
        self.info_tag = info_tag
        if self.info_tag == 'text':
            self.max_sku_desc_len = 200
            self.max_alpha_str_len = 40

            self.nkerns=[128, 128]
            self.sku_shape = (self.batch_size, self.max_alpha_str_len, 1, self.max_sku_desc_len)
            self.layer0_image_shape = self.sku_shape
            self.layer0_filter_shape = (self.nkerns[0], self.max_alpha_str_len, 1, 7)
            self.layer0_poolsize = (1, 2)
            self.layer1_image_shape = (self.batch_size, self.nkerns[0], 1, 97)
            self.layer1_filter_shape = (self.nkerns[1], self.nkerns[0], 1, 6)
            self.layer1_poolsize = (1, 2)
            self.layer2_n_in = self.nkerns[1] * 1 * 46
            self.layer2_n_out = 500
            self.layer3_n_in = self.layer2_n_out
            self.layer3_n_out = 100
        
        if self.info_tag == 'image':
            RGB = 3
            
            self.nkerns=[128, 128]
            self.sku_shape = (self.batch_size, RGB, 50, 50)
            self.layer0_image_shape = self.sku_shape
            self.layer0_filter_shape = (self.nkerns[0], RGB, 3, 3)
            self.layer0_poolsize = (2, 2)
            self.layer1_image_shape = (self.batch_size, self.nkerns[0], 24, 24)
            self.layer1_filter_shape = (self.nkerns[1], self.nkerns[0], 3, 3)
            self.layer1_poolsize = (2, 2)
            self.layer2_n_in = self.nkerns[1] * 11 * 11
            self.layer2_n_out = 1000
            self.layer3_n_in = self.layer2_n_out
            self.layer3_n_out = 100
        
        if self.info_tag == 'text,image':
            self.max_sku_desc_len = 200
            self.max_alpha_str_len = 40

            self.nkerns=[128, 128]
            self.sku_shape = (self.batch_size, 4, 50, 200)
            self.layer0_image_shape = self.sku_shape
            self.layer0_filter_shape = (self.nkerns[0], 4, 3, 9)
            self.layer0_poolsize = (2, 8)
            self.layer1_image_shape = (self.batch_size, self.nkerns[0], 24, 24)
            self.layer1_filter_shape = (self.nkerns[1], self.nkerns[0], 3, 3)
            self.layer1_poolsize = (2, 2)
            self.layer2_n_in = self.nkerns[1] * 11 * 11
            self.layer2_n_out = 1000
            self.layer3_n_in = self.layer2_n_out
            self.layer3_n_out = 100

        #重载配置
        self.reload()

#coding=utf-8
import theano
import theano.tensor as T
from cnn_conf import relu

#距离-相似度函数
def dis_cos(x1, x2):
    d = (x1* x2).sum(axis=1)
    denom= theano.tensor.sqrt((x1 * x1).sum(axis=1) * (x2 * x2).sum(axis=1))
    cos = d / denom
    return 0.5 + 0.5 * cos

def dis_euler(x1, x2):
  dist = numpy.linalg.norm(x1 - x2, axis=1)
  return 1.0 / (1.0 + dist)

#二输入模型
class PairDistanceLayer(object):
    def __init__(self, input, dis_type, cost_type):
      self.x1 = input[0]
      self.x2 = input[1]
      self.dis_type = dis_type
      self.cost_type= cost_type

    #相似度
    def sim(self, y):
        if self.dis_type == "cos":
            return dis_cos(self.x1, self.x2)
        elif self.dis_type == 'euler':
            return dis_euler(self.x1, self.x2)
        else:
            raise Exception('dis_type error')

    #cost函数
    def cost_mse(self, y):
        if self.dis_type == "cos":
            return T.mean((y - self.sim(y))**2)
        elif dis_type == "euler":
            return T.mean(linalg.norm(y - self.sim(y)))
        else:
            raise Exception('dis_type error')  

    def cost_prob(self, y):
        sim = (self.x1 * self.x2).sum(axis=1)
        prob = T.nnet.sigmoid(sim)
        return T.mean(y * T.log(prob) + (1 - y) * T.log(1 - prob))
    
    def cost(self, y):
        if self.cost_type == 'cost_prob':
            return self.cost_prob(y)
        elif self.cost_type == 'cost_mse':
            return self.cost_mse(y)
        else:
            return self.cost_mse(y)
    
class TripleDistanceLayer(PairDistanceLayer):
    def __init__(self, input, dis_type, cost_type):
      self.x1 = input[0]
      self.x2 = input[1]
      self.x3 = input[2]
      self.dis_type = dis_type
      self.cost_type= cost_type

    #相似度
    def sim(self, y):
        if self.dis_type == "cos":
            s = dis_cos(self.x1, self.x3) - dis_cos(self.x1, self.x2)
        elif self.dis_type == 'euler':
            s = dis_euler(self.x1, self.x3) - dis_euler(self.x1, self.x2)
        else:
            raise Exception('dis_type error')
        
        s *= (2 * y - 1)
        return (s + 1) / 2
    
    #优化函数
    def cost_relu(self, y):
        s = self.sim(y)
        return T.mean(relu(0.6 - s))

    def cost(self, y):
        if self.cost_type == 'cost_mse':
            return self.cost_mse(y)
        elif self.cost_type == 'cost_relu':
            return self.cost_relu(y)
        else:
            return self.cost_mse(y)


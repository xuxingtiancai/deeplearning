#!/bin/env python
#coding=utf-8

import sys
sys.path.append('./')

import numpy
import theano
from itertools import *
import os
from PIL import Image 
import sys
from lib import memo

class SkuImage(object):
  def __init__(self,image_paths):
      self.image_paths = image_paths
      #self.sku_image_dict={}
      #self.loadimage()

  #载入所有图片
  def loadimage(self):
    print >>sys.stderr, 'load image start'
    sku_image_files=os.listdir(self.image_paths)
    for i, sku_image_file in enumerate(sku_image_files, 1):
        if i % 1000 == 0:
            print >>sys.stderr, 'load image', i
        sku_id = os.path.splitext(sku_image_file)[0]
        abs_path_image = os.path.join(self.image_paths, sku_image_file)
        try:
            self.sku_image_dict[sku_id] = self.image2array(abs_path_image)
        except:
            continue
    print >>sys.stderr, 'load image success'
  
  def image2array(self, img_file):
      img = Image.open(img_file)
      test_array= (numpy.asarray(img, dtype='float64')/256)
      (x_shape,y_shape,z_shape) = test_array.shape
      aa = test_array.reshape(z_shape,y_shape,x_shape)
      return aa
  
  @memo.memo
  def fetch(self, sku_id):
       try:
           return self.image2array(os.path.join(self.image_paths, str(sku_id)+'.jpg'))
       except:
           return None

class SkuInfo(object):
  def __init__(self,input_file):
    self.sku_info_dict={}
    self.input_file = input_file
    self.parse_sku_info()
  
  def parse_sku_info(self):
    fd = open(self.input_file,'r')
    for line in fd.readlines():
      try:
        column = line.strip().split("\1")
        skuid = column[0]
        price = int(float(column[1]))
        tag = column[2]
        sku_name = column[3]
        self.sku_info_dict[skuid] = self.extract_fea(price,tag,sku_name)
      except:
        continue
    fd.close()
    print "parse sku_info success"
  
  def getSkuInfo(self,sku):
    if sku in self.sku_info_dict:
      return self.sku_info_dict[sku]
    else:
      return None
  
  def extract_fea(self,price,tag,sku_name):
    s = "price:%s,tag:%s,sku_name:%s" %(price,tag,sku_name)
    return s
    

class SkuTextReadStream(object):
  sku_info_gather = None
  sku_image_gather = None
  def __init__(self,input_file,sku_info_file_list,sku_shape,info_tag):
    self.sku_shape = sku_shape
    self.info_tag = info_tag
    self.batch_size = sku_shape[0]
    self.is_finish=False
    self.input_path = input_file
    self.fd = open(self.input_path)
    if not self.fd:
        return False

    if 'text' in self.info_tag:
        sku_info_file = sku_info_file_list.split(',')[0]
        if self.__class__.sku_info_gather == None:
            self.__class__.sku_info_gather = SkuInfo(sku_info_file)
        if self.info_tag == 'text':
            batch_size, max_alpha_str_len, one, max_sku_desc_len = sku_shape
        if self.info_tag == 'text,image':
            batch_size, one, max_alpha_str_len, max_sku_desc_len = sku_shape
        self.max_alpha_str_len = max_alpha_str_len
        self.max_sku_desc_len = max_sku_desc_len
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,-:"[:self.max_alpha_str_len-1]
        self.encode_dict=dict((v, k) for k, v in enumerate(self.alphabet, 1))

    if 'image' in self.info_tag:
        sku_info_file = sku_info_file_list.split(',')[-1]
        if self.__class__.sku_image_gather == None:
            self.__class__.sku_image_gather = SkuImage(sku_info_file)
  
  def read_data(self, sku_num):
    self.label_column = []
    self.sku_id_columns = tuple([] for i in range(sku_num))
    self.sku_image_columns = tuple([] for i in range(sku_num))

    #读入一个batch的数据
    num = 0
    while True:
        line = self.fd.readline()
        if not line:
            self.is_finish = True
            self.fd.close()
            break

        if self.parse_line(line, sku_num) == None:
            continue
        num += 1
        if num == self.batch_size:
            break
    
    return self.covert_column(self.label_column, 'y'), tuple(self.covert_column(c, 'x') for c in self.sku_image_columns), self.sku_id_columns
    

  def parse_line(self, line, sku_num):
    items = line.strip().split('\t')
    if len(items) != sku_num + 1:
        return None

    label, sku_ids = items[0], items[1:]
    cells = []
    for sku_id in sku_ids:
        cell = self.convert_cell(sku_id)
        if cell == None:
            return None
        cells.append(cell)

    self.label_column.append(int(label))
    for sku_image_column, cell, sku_id_column, sku_id in izip(self.sku_image_columns, cells, self.sku_id_columns, sku_ids):
        sku_image_column.append(cell)
        sku_id_column.append(sku_id)
    
    return 0
  
  #单元格级别的转化
  #文字信息
  def sku2text(self, sku_id):
    sku = self.sku_info_gather.getSkuInfo(sku_id)
    if sku == None:
        return None

    sku_image = numpy.zeros((self.max_alpha_str_len, self.max_sku_desc_len),dtype=int)
    no_zeros_list = [ self.encode_dict.get(s,0) for s in sku]
    for i in xrange(0,min(len(no_zeros_list),self.max_sku_desc_len)):
        sku_image[no_zeros_list[i],i] = 1
    return sku_image
  
  #图片信息
  def sku2image(self, sku_id):
    sku_image = self.sku_image_gather.fetch(sku_id)
    
    if self.info_tag == 'image':
        if sku_image == None or sku_image.shape != self.sku_shape[1:]:
            return None
    return sku_image

  #sku_id -> 特征
  def convert_cell(self, sku_id):
    if self.info_tag == 'text':
        return self.sku2text(sku_id)
    if self.info_tag == 'image':
        return self.sku2image(sku_id)
    if self.info_tag == 'text,image':
        text = self.sku2text(sku_id)
        if text == None:
            return None
        image = self.sku2image(sku_id)
        if image == None:
            return None
        
        merge_sku = numpy.zeros(tuple(self.sku_shape[1:]))
        try:
            for i in range(len(text)):
                for j in range(len(text[0])):
                    merge_sku[0][i][j] = text[i][j]
            for i in range(len(image)):
                for j in range(len(image[0])):
                    for k in range(len(image[0][0])):
                        merge_sku[i + 1][j][k] = image[i][j][k]
        except Exception, e:
            print >>sys.stderr, 'sku_id=', sku_id 
            print >>sys.stderr, 'image=', len(image), len(image[0]), len(image[0][0])
            print >>sys.stderr, e
        return merge_sku

  #column级别的转化
  def covert_column(self, column, tag):
    #debug
    mat = numpy.asarray(column, dtype=theano.config.floatX)
    
    if tag == 'x':
        mat.shape = (len(column),) + self.sku_shape[1:]
    if tag == 'y':
        mat.shape = len(column)
    #shared
    return theano.shared(mat, borrow=True)

if __name__ == "__main__":
  import sys
  train_file = sys.argv[1]
  sku_file = sys.argv[2]
  info_tag = sys.argv[3]
  if info_tag == 'text':
      sku_shape = (2, 40, 1, 200)
  if info_tag == 'image':  
      sku_shape = (256, 3, 50, 50)
  if info_tag == 'text,image':
      sku_shape = (2, 4, 50, 200)

  train_stream = SkuTextReadStream(train_file,sku_file,sku_shape,info_tag)
  while not train_stream.is_finish:
      x = train_stream.read_data(2)[1]
      t = x[0].get_value()
      print 't', t.shape
      ins = t[0]
      print 'ins', ins.shape
      for i in range(len(ins)):
          print i, ins[i].sum()

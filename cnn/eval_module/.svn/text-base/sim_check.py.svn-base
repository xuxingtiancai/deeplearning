#!/user/bin/env python

import sys
import redis
sys.path.append('../')
from proto import AdSkuAttribute_pb2

reload(sys)
sys.setdefaultencoding( "utf-8" )
input = sys.argv[1]

host="127.0.0.1"
port=22139
pool = redis.ConnectionPool(host=host, port=port,db=0)
goods_redis = redis.StrictRedis(connection_pool=pool) 

for line in open(input,'r'):
    line_list = line.strip().split()
    num = len(line_list)
    sku_list = [""]*num
    ad_sku_attribute = AdSkuAttribute_pb2.AdSkuAttribute()
    sku_info = goods_redis.get(line_list[0])
    if not sku_info:
        continue
    ad_sku_attribute.ParseFromString(sku_info)
    main_pid = ad_sku_attribute.pid
    sku_id = ad_sku_attribute.sku_id
    sku_list[0] = (line_list[0],1)
    for i in xrange(1,num):
        sku_list[i] = line_list[i].split(':')
    sku_name_list = []
    sku_info_list = []
    #try:
    for i in xrange(num):
        sku_info = goods_redis.get(sku_list[i][0])
        sku_tuple = (sku_info,sku_list[i][1])
        sku_info_list.append(sku_tuple)
    pid_set = set()
    for i in xrange(len(sku_info_list)):
        if not sku_info_list[i][0]:
            continue
        if float(sku_info_list[i][1])< 0.7:
            continue
        ad_sku_attribute = AdSkuAttribute_pb2.AdSkuAttribute()
        ad_sku_attribute.ParseFromString(sku_info_list[i][0])
        pid = ad_sku_attribute.pid
        if pid in pid_set or (pid == main_pid and i!=0):
            continue
        pid_set.add(pid)
        sku_name_list.append((ad_sku_attribute.sku_name,ad_sku_attribute.sku_id,pid,sku_info_list[i][1]))
    for sku in sku_name_list:
        print sku[0],sku[1],sku[2],sku[3]
    print '\n'
    #except Exception,e:
    #    print e
    #    continue


#triple-输入数据
#train_data=data_input/triple/train_data
#eval_data=data_input/triple/eval_data
#sku_num=3
#model=data_model/triple/model.pickle

#pair-输入数据
#head -10000000 data_input/pair/train_new > data_input/pair/train_new.head
#train_data=data_input/pair/train_new.head
train_data=data_input/pair/train_new
eval_data=data_input/pair/eval_new
sku_num=2
model=data_model/pair/model.pickle
sav_model=data_model/pair/sav.model.pickle

#sku_info
#info_tag=text
#sku_info=data_input/view_sku_info

#info_tag=image
#sku_info=/data1/guowentao/theano/sku_pair_flat/data_input/sku_picture

info_tag=text,image
sku_info=data_input/view_sku_info,/data1/guowentao/theano/sku_pair_flat/data_input/sku_picture

#训练模型
/bin/rm -rf ${model}*
THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True' python sku_cnn.py $train_data $eval_data $model $sku_info $sku_num $info_tag 

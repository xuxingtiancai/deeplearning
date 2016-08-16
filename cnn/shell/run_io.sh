
head -10000 data_input/pair/train_new > data_input/pair/train_new.head
train_data=data_input/pair/train_new.head
#train_data=data_input/pair/train_new

#sku_info=/data1/guowentao/theano/sku_pair_flat/data_input/test_image_input
#sku_info=/data1/guowentao/theano/sku_pair_flat/data_input/sku_picture
#info_tag=image

#sku_info=data_input/view_sku_info
#info_tag=text

sku_info=data_input/view_sku_info,/data1/guowentao/theano/sku_pair_flat/data_input/sku_picture
info_tag=text,image

python io_module/sku_io.py $train_data $sku_info $info_tag

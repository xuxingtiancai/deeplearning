
model=data_model/pair/model.pickle-35154

info_tag=image
sku_info=/data1/guowentao/theano/sku_pair_flat/data_input/sku_picture
sku_num=2

model_input=data_input/model_sku
candidate_input=data_input/candidate_sku
python io_module/vec_io.py $model_input $model data_output/model.vec $sku_info $sku_num $info_tag
python io_module/vec_io.py $candidate_input $model data_output/candidate.vec $sku_info $sku_num $info_tag

#sh -x shell/run_eval.sh
scp data_output/{model.vec,candidate.vec} admin@172.17.39.149:/export/App/xuxing_files

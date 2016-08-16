


function add_title() {
  num=$(wc -l $1 | awk '{print $1}')
  size=$2
  echo "$num $size" > $3
  cat $1 >> $3
}

model_file=data_output/model.vec
candidate_file=data_output/candidate.vec
vec_size=$(tail -1 $candidate_file | awk '{print NF-1}')
add_title $candidate_file $vec_size ${candidate_file}.title

output_file=data_eval/cnn.eval
/bin/rm -rf ${output_file}*

#旧的cal_sim
cd eval_module
rm cal_sim_stream
g++ -o cal_sim_stream -lpthread cal_sim_stream.cpp
cd ..
chmod +x eval_module/cal_sim_stream
cat $model_file | eval_module/cal_sim_stream -ad_sku_vec_file ${candidate_file}.title -thread 10 -top_num 20 > $output_file

#新的cal_sim
#head -10 $model_file | eval_module/cal_sim_stream
#-candidate_sku_vec_file $candidate_file
#-thread  6 
#-top_num 20 
#-openmp 0 
#-candidate_sku_vec_num 875630   
#-sku_vec_size 100 
#-sku_category_file data/skuid_categoryid.txt  
#-category_index 3 



lr=5e-5
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=256
epochs=30

load_model_path=/home/dxx/DHPNpr/saved_models/UniXcoder_256/checkpoint-best-ppl/pytorch_model.bin
output_dir=/home/dxx/DHPNpr/data_dis/UniXcoder
log_file=UniXcoder_data_dis.log
train_file=/home/dxx/DHPNpr/data/Train
validate_file=/home/dxx/DHPNpr/data/Valid
model_name_or_path=/home/dxx/DHPNpr/UniXcoder/unixcoder-base
tokenizer_name=/home/dxx/DHPNpr/UniXcoder/unixcoder-base
log_file_dir=/home/dxx/DHPNpr/logging

mkdir -p $output_dir

python ./run.py \
--do_test \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--load_model_path $load_model_path \
--train_dir $train_file \
--dev_dir $validate_file \
--output_dir $output_dir \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs \
--log_file_dir $log_file_dir \
2>&1| tee $output_dir/$log_file
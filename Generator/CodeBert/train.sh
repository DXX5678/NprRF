
lr=5e-5
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=512
train_steps=400000
eval_steps=3000


output_dir=/home/dxx/DHPNpr/saved_models/CodeBert
log_file=train.log
train_file=/home/dxx/DHPNpr/data/Train
validate_file=/home/dxx/DHPNpr/data/Valid
model_name_or_path=/home/dxx/DHPNpr/CodeBert/codebert-base
tokenizer_name=/home/dxx/DHPNpr/CodeBert/codebert-base
log_file_dir=/home/dxx/DHPNpr/logging

mkdir -p $output_dir

python ./run.py \
--do_train \
--do_eval \
--model_type roberta \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--train_dir $train_file \
--dev_dir $validate_file \
--output_dir $output_dir \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--train_steps $train_steps \
--eval_steps $eval_steps \
--log_file_dir $log_file_dir \
2>&1| tee $output_dir/$log_file
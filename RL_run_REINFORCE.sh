# APPT
reward=APPT
model_path=/home/dingxuxing/NPRwRL/Reward/APPT/bert-base-uncased
# model_load_path=/home/dingxuxing/NPRwRL/Reward/APPT/bert_small_cat_headTail_0/checkpoint
model_load_path=/home/dingxuxing/NPRwRL/Reward/APPT/save_model/CodeT5/checkpoint6
max_length=512

# CodeBERT
# generator=CodeBERT
# model_type=roberta
# model_name_or_path=/home/dingxuxing/NPRwRL/Generator/CodeBert/codebert-base
# load_model_path=/home/dingxuxing/NPRwRL/Generator/saved_models/CodeBert/checkpoint-best-ppl/pytorch_model.bin
# tokenizer_name=/home/dingxuxing/NPRwRL/Generator/CodeBert/codebert-base
# max_source_length=512
# max_target_length=512
# beam_size=1

# CodeT5
generator=CodeT5
model_type=codet5
model_name_or_path=/home/dingxuxing/NPRwRL/Generator/CodeT5/codet5-base
load_model_path=/home/dingxuxing/NPRwRL/Generator/saved_models/CodeT5_256/checkpoint-best-ppl/pytorch_model.bin
tokenizer_name=/home/dingxuxing/NPRwRL/Generator/CodeT5/codet5-base
max_source_length=512
max_target_length=256
beam_size=5

# UniXcoder
# generator=UniXcoder
# model_type=unixcoder
# model_name_or_path=/home/dingxuxing/NPRwRL/Generator/UniXcoder/unixcoder-base
# load_model_path=/home/dingxuxing/NPRwRL/Generator/saved_models/UniXcoder_256/checkpoint-best-ppl/pytorch_model.bin
# tokenizer_name=/home/dingxuxing/NPRwRL/Generator/UniXcoder/unixcoder-base
# max_source_length=512
# max_target_length=256
# beam_size=1

train_file=/home/dingxuxing/NPRwRL/data/Train
validate_file=/home/dingxuxing/NPRwRL/data/Valid
output_dir=/home/dingxuxing/NPRwRL/Generator/saved_models/REINFORCE_${reward}_${generator}
log_file_dir=/home/dingxuxing/NPRwRL/logging/REINFORCE_${reward}_${generator}
log_file=train.log
epoch=30
lr=5e-5
batch_size=8 #32
reward_smooth_alpha=0.9

mkdir -p $output_dir
mkdir -p $log_file_dir

python -m RL.REINFORCE.train \
--generator $generator \
--model_type $model_type \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--load_model_path $load_model_path \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--reward $reward \
--model_path $model_path \
--model_load_path $model_load_path \
--reward_max_length $max_length \
--train_dir $validate_file \
--dev_dir $train_file \
--output_dir $output_dir \
--log_file_dir $log_file_dir \
--num_train_epochs $epoch \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--reward_smooth_alpha $reward_smooth_alpha \
2>&1| tee $log_file_dir/$log_file
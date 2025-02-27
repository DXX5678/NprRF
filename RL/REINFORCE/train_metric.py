import math
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from RL.REINFORCE.REINFORCE import REINFORCE, RewardBuffer
from RL.REINFORCE.configs import Config
from RL.preprocess_Data import load_gen_data
from Reward.Metric.main import get_score


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


# 定义损失函数
def compute_rl_loss(logits, actions, rewards):
    """
    logits: 生成器输出的概率分布 (batch_size, seq_len, vocab_size)
    actions: 生成的 patch (batch_size, seq_len)
    rewards: 评估模型给出的奖励 (batch_size)
    """
    log_probs = torch.gather(logits, -1, actions.unsqueeze(-1)).squeeze(-1)
    log_probs = log_probs.mean(dim=1)  # 平均所有token的概率
    return -(log_probs * rewards).mean()  # REINFORCE损失


def eval_reward_epoch(args, configs, eval_data, eval_examples, reinforce):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    reinforce.eval()
    reward_value, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval reward"):
        start_idx = batch_num * args.eval_batch_size
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask, target_ids, target_mask, buggy_ids, buggy_mask = batch
        with torch.no_grad():
            # 生成补丁
            _, patch_out, preds = reinforce(source_ids, source_mask, target_ids, target_mask, args.generator)
            
            pred_nls = []
            if args.generator=='CodeBERT':
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = configs.generator[2].decode(t, clean_up_tokenization_spaces=False)
                    pred_nls.append(text)
            elif args.generator=='CodeT5':
                for pred in preds:
                    top_p = pred.cpu().numpy()
                    top_p = list(top_p)
                    text = configs.generator[2].decode(top_p, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_nls.append(text.replace("<FIXS>", "").replace("<FIXE>", ""))
            else:
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = configs.generator[2].decode(t, clean_up_tokenization_spaces=False)
                    pred_nls.append(text.replace("<FIXS>", "").replace("<FIXE>", ""))    

            # 计算奖励
            # for text in pred_nls:
            #     logger.info(text)
            reward = []
            for i, pl in enumerate(pred_nls):
                idx = start_idx + i
                example = eval_examples[idx]
                buggy_method = example.buggy_method
                ref = buggy_method.split('\n')
                pre = buggy_method.split('\n')
                for ind in range(len(ref)):
                    if example.buggy_lines in ref[ind]:
                        ref[ind] = example.fix_lines
                        pre[ind] = pl
                reward.append(get_score(buggy=buggy_method, reference='\n'.join(ref), prediction='\n'.join(pre), model_type=args.model_path))
            smoothed_reward = reward_buffer.smooth(np.array(reward))

        reward_value += np.mean(smoothed_reward)
        batch_num += 1
    reward_value = reward_value / batch_num
    reinforce.train()
    return round(reward_value, 3)


if __name__ == "__main__":
    # 载入命令行参数
    parser = argparse.ArgumentParser()
    # 生成器加载参数
    parser.add_argument("--generator", type=str, required=True,
                        help="Select one pre-train model.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")

    # 奖励模型加载参数
    parser.add_argument("--reward", type=str, required=True,
                        help="Select one APPC model.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    
    # RL训练参数
    parser.add_argument("--train_dir", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_dir", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file_dir", default=None, type=str, required=True,
                        help="The output directory where the log_file will be written.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--reward_smooth_alpha", default=0.9, type=float,
                        help="奖励平滑的指数加权衰减因子")
    parser.add_argument("--patience", default=3, type=int)
    args = parser.parse_args()
    logger.info(args)
    
    t0 = time.time()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    # make dir if output_dir/log_file_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    if os.path.exists(args.log_file_dir) is False:
        os.makedirs(args.log_file_dir)

    # 加载配置类
    configs = Config(args)
    
    # 加载REINFORCE模型
    reinforce = REINFORCE(configs.generator[1], configs.generator[2], configs.reward[1])
    reinforce.to(args.device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        preinforcepo = DDP(reinforce)
    elif args.n_gpu > 1:
        # for DataParallel
        reinforce = torch.nn.DataParallel(reinforce)
    log_file = open(os.path.join(args.log_file_dir, 'RL.log'), 'a+')

    # 数据预处理：1. 生成器模型数据 2. 奖励模型数据
    train_examples, train_data = load_gen_data(args, configs, args.train_dir, mode="val")
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      shuffle=False,
                                      drop_last=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in reinforce.module.generator_model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in reinforce.module.generator_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)
    reward_buffer = RewardBuffer(args)
    # Start training
    train_example_num = len(train_data)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_example_num)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Batch num = %d", len(train_dataloader))
    logger.info("  Num epoch = %d", args.num_train_epochs)
    log_file.write("  Start Training...\n")
    global_step, best_avg_reward = 0, 0
    dev_dataset = {}
    for cur_epoch in range(int(args.num_train_epochs)):
        logger.info("  Now epoch = %d", cur_epoch)
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        reinforce.train()
        for step, batch in enumerate(bar):
            start_idx = nb_tr_steps * (args.train_batch_size // args.gradient_accumulation_steps)
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, buggy_ids, buggy_mask = batch
            # 生成补丁
            logits, patch_out, preds = reinforce(source_ids, source_mask, target_ids, target_mask, args.generator)
            # logger.info(preds.shape)
            # logger.info(preds.data)
            actions = patch_out[:, 1:]
            pred_nls = []
            if args.generator=='CodeBERT':
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = configs.generator[2].decode(t, clean_up_tokenization_spaces=False)
                    pred_nls.append(text)
            elif args.generator=='CodeT5':
                for pred in preds:
                    top_p = pred.cpu().numpy()
                    top_p = list(top_p)
                    text = configs.generator[2].decode(top_p, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    # logger.info(text)
                    pred_nls.append(text.replace("<FIXS>", "").replace("<FIXE>", ""))
            else:
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = configs.generator[2].decode(t, clean_up_tokenization_spaces=False)
                    pred_nls.append(text.replace("<FIXS>", "").replace("<FIXE>", ""))              

            # 计算奖励
            # for text in pred_nls:
            #     logger.info(text)
            reward = []
            for i, pl in enumerate(pred_nls):
                idx = start_idx + i
                example = train_examples[idx]
                buggy_method = example.buggy_method
                ref = buggy_method.split('\n')
                pre = buggy_method.split('\n')
                for ind in range(len(ref)):
                    if example.buggy_lines in ref[ind]:
                        ref[ind] = example.fix_lines
                        pre[ind] = pl
                # logger.info('\n'.join(pre))
                reward.append(get_score(buggy=buggy_method, reference='\n'.join(ref), prediction='\n'.join(pre), model_type=args.model_path))
            smoothed_reward = reward_buffer.smooth(np.array(reward))
            rewards = torch.tensor(smoothed_reward).to(device)

            # 计算REINFORCE损失
            loss = compute_rl_loss(logits, actions, rewards)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1

            # 反向传播和优化
            loss.backward(retain_graph=True)
            if nb_tr_steps % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                log_file.write("[{}] Train loss {}\n".format(cur_epoch, round(train_loss, 3)))

        
        # Eval model with dev dataset
        logger.info("  Start Eval...")
        log_file.write("  Start Eval...\n")
        if 'dev_loss' in dev_dataset:
            eval_examples, eval_data = dev_dataset['dev_loss']
        else:
            eval_examples, eval_data = load_gen_data(args, configs, args.dev_dir, mode="train", number=10000)
            dev_dataset['dev_loss'] = eval_examples, eval_data

        eval_reward = eval_reward_epoch(args, configs, eval_data, eval_examples, reinforce)
        result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_reward': eval_reward}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            log_file.write("  %s = %s\n" % (key, str(result[key])))
        logger.info("  " + "*" * 20)
        log_file.write("  End Eval...\n")
        # save last checkpoint
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = reinforce.module.generator_model if hasattr(reinforce, 'module') else reinforce.generator_model
        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Save the last model into %s", output_model_file)

        if eval_reward > best_avg_reward:
            not_reward_asc_cnt = 0
            logger.info("  Best reward:%s", eval_reward)
            logger.info("  " + "*" * 20)
            log_file.write("[%d] Best reward changed into %.4f\n" % (cur_epoch, eval_reward))
            best_avg_reward = eval_reward

            # Save best checkpoint for best reward
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-reward')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = reinforce.module.generator_model if hasattr(reinforce, 'module') else reinforce.generator_model
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            logger.info("Save the best reward model into %s", output_model_file)
        else:
            not_reward_asc_cnt += 1
            logger.info("reward does not ascend for %d epochs", not_reward_asc_cnt)
            if any([x >= args.patience for x in [not_reward_asc_cnt]]):
                early_stop_str = "[%d] Early stop as not_reward_asc_cnt=%d\n" % (cur_epoch, not_reward_asc_cnt)
                logger.info(early_stop_str)
                log_file.write(early_stop_str)
                break
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    logger.info("Finish training and take %s", get_elapse_time(t0))

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import math
import os
import logging
import argparse

import numpy as np
from tqdm import tqdm
import time
import torch
import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import data_preprocess
from configs import set_dist, set_seed
from model import build_or_load_gen_model
from utils import get_elapse_time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                            labels=target_ids, decoder_attention_mask=target_mask)
            loss = outputs.loss
        # eval_loss += loss.item()
        eval_loss += torch.mean(loss).item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, default='refine',
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", default='save', action='store_true')
    parser.add_argument("--always_save_model", default='save', action='store_true')
    parser.add_argument("--do_eval_bleu", default='save', action='store_true',
                        help="Whether to evaluate bleu on dev set.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file_dir", default=None, type=str, required=True,
                        help="The output directory where the log_file will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_dir", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_dir", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--checkpoint_model_type", default=None, type=str,
                        help="The checkpoint_model_type in testing, file_type: best-bleu/ppl/last.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to inference.")

    parser.add_argument("--do_defects4j", action='store_true',
                        help="Whether to inference on the Defect4J.")
    parser.add_argument("--buggy_file", default="", type=str,
                        help="Path of the buggy project on the Defect4J.")
    parser.add_argument("--buggy_line", default="", type=str,
                        help="Location of the buggy code.")
    parser.add_argument("--start_line", default=-1, type=int,
                        help="Location of the buggy method.")
    parser.add_argument("--end_line", default=-1, type=int,
                        help="Location of the buggy method.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")

    # print arguments
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

    # Set seed
    set_dist(args)
    set_seed(args)

    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    # make dir if cache_path not exist
    if os.path.exists(args.cache_path) is False:
        os.makedirs(args.cache_path)

    config, model, tokenizer = build_or_load_gen_model(args)
    # tokenizer.add_tokens(["<BUGS>", "<BUGE>", "<FIXS>", "<FIXE>"])
    model.to(args.device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)

    pool = multiprocessing.Pool(args.cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    log_file = open(os.path.join(args.log_file_dir, 'CodeT5.log'), 'a+')

    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = data_preprocess.load_and_cache_gen_data(args, args.train_dir, pool, tokenizer,
                                                                             'train', mode="train")
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)
        log_file.write("  Start Training...\n")
        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info("  Now epoch = %d", cur_epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                    log_file.write("[{}] Train loss {}\n".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval and (int(cur_epoch) % 1 == 0):
                # Eval model with dev dataset
                logger.info("  Start Eval...")
                log_file.write("  Start Eval...\n")
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = data_preprocess.load_and_cache_gen_data(args, args.dev_dir, pool,
                                                                                       tokenizer, 'dev', mode='dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    log_file.write("  %s = %s\n" % (key, str(result[key])))
                logger.info("  " + "*" * 20)
                log_file.write("  End Eval...\n")
                # if args.data_num == -1:
                #     tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    log_file.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if any([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        log_file.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()

                """
                if args.do_eval_bleu:
                    eval_examples, eval_data = data_preprocess.load_and_cache_gen_data(args, args.dev_dir, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'eval')
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    elif args.task in ['refine']:
                        dev_bleu_em = dev_bleu
                    else:
                        dev_bleu_em = dev_bleu + dev_em

                    # if args.data_num == -1:
                    #     tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        # fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                        #     cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        # fa.write(
                        #     "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                        #         cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            # fa.write(stop_early_str)
                            break
                """
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        # if args.local_rank in [-1, 0] and args.data_num == -1:
        #     tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        change = False
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            change = True
        logger.info("***** Running testing *****")
        if args.do_defects4j:
            logger.info("***** Defects4J *****")
            files = []
            if args.dev_dir is not None:
                files.append(args.dev_dir)
            if args.test_filename is not None:
                files.append(args.test_filename)
            for idx, file in enumerate(files):
                logger.info("Test file: {}".format(file))
                test_examples, test_data = data_preprocess.load_and_cache_gen_test_data(args, args.test_filename, pool,
                                                                                        tokenizer, "test")
                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                # pred_f = codecs.open(args.test_filename + ".output", 'w', encoding='utf8')
                for batch in tqdm(test_dataloader, total=len(test_dataloader)):
                    source_ids = batch[0].to(args.device)
                    source_mask = source_ids.ne(tokenizer.pad_token_id)
                    with torch.no_grad():
                        preds = model.generate(source_ids,
                                               attention_mask=source_mask,
                                               use_cache=True,
                                               num_beams=args.beam_size,
                                               num_return_sequences=args.beam_size,
                                               early_stopping=args.task == 'refine',
                                               max_length=args.max_target_length)
                        # print("preds length",len(preds))
                        for nb_pred in preds:
                            t = nb_pred.cpu().numpy()
                            t = list(t)
                            text = tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                            # print(text)
                            p.append(text.replace("<FIXS>", "").replace("<FIXE>", ""))
                            # text = "<PRED_START>" + text.strip() + "<PRED_END>"
                            # pred_f.write(text + '\n')

                buggy_file_lines = open(args.buggy_file, "r").readlines()
                buggy_line_number = list(map(int, args.buggy_line.split(",")))
                for i in range(len(p)):
                    output_file = os.path.join(args.output_dir, str(i + 1), os.path.basename(args.buggy_file))
                    os.makedirs(os.path.dirname(output_file))
                    output_file = open(output_file, "w")
                    flag = False
                    for j in range(len(buggy_file_lines)):
                        if (j + 1) in buggy_line_number:
                            if not flag:
                                buggy_line = buggy_file_lines[j]
                                white_space_before_buggy_line = buggy_line[0:buggy_line.find(buggy_line.lstrip())]
                                output_file.write(white_space_before_buggy_line + p[i].strip() + "\n")
                                flag = True
                            else:
                                continue
                        else:
                            output_file.write(buggy_file_lines[j])
                    output_file.close()
        else:
            # Build the dataset for discriminator
            for_train_examples, for_train_data = data_preprocess.load_and_cache_gen_data(args, args.train_dir, pool,
                                                                                         tokenizer,
                                                                                         'test', mode="train")
            sampler = SequentialSampler(for_train_data)
            dataloader = DataLoader(for_train_data, sampler=sampler, batch_size=args.train_batch_size)

            logger.info("***** Start building train dataset*****")
            model.eval()
            p = []
            for batch in tqdm(dataloader):
                source_ids = batch[0].to(args.device)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                with torch.no_grad():
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=args.beam_size,
                                           early_stopping=args.task == 'refine',
                                           max_length=args.max_target_length)
                    top_preds = list(preds.cpu().numpy())
                    p.extend(top_preds)
            pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                        p]
            ids = data_preprocess.readLines(os.path.join(args.train_dir, "trn.ids"))
            patch_lines_dir = os.path.join(os.path.join(args.output_dir, "Train"), "patch_lines")
            os.makedirs(patch_lines_dir)
            index = 0
            assert len(ids) == len(for_train_examples)
            for id in tqdm(ids):
                ref = pred_nls[index].replace("<FIXS>", "").replace("<FIXE>", "")
                gold = for_train_examples[index]
                if ref.replace(' ', '') != gold.target.replace("<FIXS>", "").replace("<FIXE>", "").replace(' ', ''):
                    with open(os.path.join(patch_lines_dir, id + ".txt"), 'w') as f:
                        f.write(ref + '\n')
                    f.close()
                index += 1
            logger.info("***** End building train dataset*****")

            for_valid_examples, for_valid_data = data_preprocess.load_and_cache_gen_data(args, args.dev_dir, pool,
                                                                                         tokenizer,
                                                                                         'test', mode="valid")
            sampler = SequentialSampler(for_valid_data)
            dataloader = DataLoader(for_valid_data, sampler=sampler, batch_size=args.train_batch_size)

            logger.info("***** Start building valid dataset*****")
            model.eval()
            p = []
            for batch in tqdm(dataloader):
                source_ids = batch[0].to(args.device)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                with torch.no_grad():
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=args.beam_size,
                                           early_stopping=args.task == 'refine',
                                           max_length=args.max_target_length)
                    top_preds = list(preds.cpu().numpy())
                    p.extend(top_preds)
            pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                        p]

            ids = data_preprocess.readLines(os.path.join(args.dev_dir, "valid.ids"))
            patch_lines_dir = os.path.join(os.path.join(args.output_dir, "Valid"), "patch_lines")
            os.makedirs(patch_lines_dir)
            index = 0
            assert len(ids) == len(for_valid_examples)
            for id in tqdm(ids):
                ref = pred_nls[index].replace("<FIXS>", "").replace("<FIXE>", "")
                gold = for_valid_examples[index]
                if ref.replace(' ', '') != gold.target.replace("<FIXS>", "").replace("<FIXE>", "").replace(' ', ''):
                    with open(os.path.join(patch_lines_dir, id + ".txt"), 'w') as f:
                        f.write(ref + '\n')
                    f.close()
                index += 1
            logger.info("***** End building valid dataset*****")
        if change:
            model = torch.nn.DataParallel(model)
        # for criteria in ['best-bleu']:
        #     criteria = args.checkpoint_model_type  # best-bleu best-ppl best-last
        #     file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
        #     logger.info("Reload model from {}".format(file))
        #
        #     if hasattr(model, 'module'):
        #         model.module.load_state_dict(torch.load(file))
        #     else:
        #         model.load_state_dict(torch.load(file))
        #     eval_examples, eval_data = data_preprocess.load_and_cache_gen_data(args, args.test_filename, pool,
        #                                                                        tokenizer, 'test',
        #                                                                        only_src=True, is_sample=False)
        #     result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
        #     test_bleu, test_em = result['bleu'], result['em']
        #     # test_codebleu = result['codebleu'] if 'codebleu' in result else 0
        #     result_str = "[%s] bleu-4: %.2f, em: %.4f\n" % (criteria, test_bleu, test_em)
        #     logger.info(result_str)
        #     # fa.write(result_str)
        #     if args.res_fn:
        #         with open(args.res_fn, 'a+') as f:
        #             f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
        #             f.write(result_str)

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    log_file.write("Finish and take {}\n".format(get_elapse_time(t0)))
    log_file.close()
    # logger.info("  " + "Test End" + "*" * 20)


if __name__ == "__main__":
    main()

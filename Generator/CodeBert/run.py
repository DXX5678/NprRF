import os
import random
import logging
import argparse
import time

import numpy as np
from io import open
from itertools import cycle

import torch
import torch.nn as nn

from utils import get_elapse_time
from data_preprocess import convert_examples_to_features, load_gen_data, read_test_examples, load_gen_data_test, \
    readLines
from model import build_or_load_gen_model
from configs import set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default="CodeBERT-finetune")
    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
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
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
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
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    t0 = time.time()
    logger.info(args)

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
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config, model, tokenizer = build_or_load_gen_model(args)

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)
    log_file = open(os.path.join(args.log_file_dir, 'CodeBert.log'), 'a+')

    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = load_gen_data(args, tokenizer, args.train_dir, mode="train")

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_examples))
        log_file.write("  Start Training...\n")
        model.train()
        dev_dataset = {}
        not_loss_dec_cnt = 0
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        bar = tqdm(range(num_train_optimization_steps), total=num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        eval_flag = True
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                  target_mask=target_mask)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("loss {}".format(train_loss))
            log_file.write("loss {}\n".format(train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                eval_flag = True

            if args.do_eval and ((global_step + 1) % args.eval_steps == 0) and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_gen_data(args, tokenizer, args.dev_dir, mode="valid")
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                log_file.write("  Start Eval...\n")

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num, _ = model(source_ids=source_ids, source_mask=source_mask,
                                                target_ids=target_ids, target_mask=target_mask)
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    log_file.write("  %s = %s\n" % (key, str(result[key])))
                logger.info("  " + "*" * 20)
                log_file.write("  End Eval...")

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("Save the last model into %s", output_model_file)

                if eval_loss < best_loss:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    log_file.write("  Best ppl:%s\n" % round(np.exp(eval_loss), 5))
                    best_loss = eval_loss

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_loss_dec_cnt]]):
                        early_stop_str = "Early stop as not_loss_dec_cnt=%d\n" % not_loss_dec_cnt
                        logger.info(early_stop_str)
                        log_file.write(early_stop_str)
                        break

                """  
                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    eval_examples = read_APRexamples(args.dev_dir)
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
                    eval_data = TensorDataset(all_source_ids,all_source_mask)   
                    dev_dataset['dev_bleu']=eval_examples,eval_data



                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval() 
                p=[]
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask= batch                  
                    with torch.no_grad():
                        preds = model(source_ids=source_ids,source_mask=source_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions=[]
                with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                    for ref,gold in zip(p,eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
                dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  "+"*"*20)    
                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               """
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
                test_examples = read_test_examples(file)
                test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
                all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
                all_source_mask = torch.tensor([f.source_mask for f in test_features], dtype=torch.long)
                test_data = TensorDataset(all_source_ids, all_source_mask)

                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                # pred_f = codecs.open(args.test_filename + ".output", 'w', encoding='utf8')
                for batch in tqdm(test_dataloader, total=len(test_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        # print("preds length",len(preds))
                        for pred in preds:
                            for nb_pred in pred:
                                t = nb_pred.cpu().numpy()
                                t = list(t)
                                if 0 in t:
                                    t = t[:t.index(0)]
                                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                                # print(text)
                                p.append(text)
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
                                output_file.write(white_space_before_buggy_line + p[i] + "\n")
                                flag = True
                            else:
                                continue
                        else:
                            output_file.write(buggy_file_lines[j])
                    output_file.close()
        else:
            # Build the dataset for discriminator
            for_train_examples, for_train_data = load_gen_data_test(args, tokenizer, args.train_dir, target="train")
            sampler = SequentialSampler(for_train_data)
            dataloader = DataLoader(for_train_data, sampler=sampler, batch_size=args.train_batch_size)

            logger.info("***** Start building train dataset*****")
            model.eval()
            p = []
            for batch in tqdm(dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

            ids = readLines(os.path.join(args.train_dir, "trn.ids"))
            patch_lines_dir = os.path.join(os.path.join(args.output_dir, "Train"), "patch_lines")
            os.makedirs(patch_lines_dir)
            index = 0
            assert len(ids) == len(for_train_examples)
            for id in tqdm(ids):
                ref = p[index]
                gold = for_train_examples[index]
                if ref.replace(' ', '') != gold.target.replace(' ', ''):
                    with open(os.path.join(patch_lines_dir, id + ".txt"), 'w') as f:
                        f.write(ref + '\n')
                    f.close()
                index += 1
            logger.info("***** End building train dataset*****")

            for_valid_examples, for_valid_data = load_gen_data_test(args, tokenizer, args.dev_dir, target="valid")
            sampler = SequentialSampler(for_valid_data)
            dataloader = DataLoader(for_valid_data, sampler=sampler, batch_size=args.train_batch_size)

            logger.info("***** Start building valid dataset*****")
            model.eval()
            p = []
            for batch in tqdm(dataloader):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)

            ids = readLines(os.path.join(args.dev_dir, "valid.ids"))
            patch_lines_dir = os.path.join(os.path.join(args.output_dir, "Valid"), "patch_lines")
            os.makedirs(patch_lines_dir)
            index = 0
            assert len(ids) == len(for_valid_examples)
            for id in tqdm(ids):
                ref = p[index]
                gold = for_valid_examples[index]
                if ref.replace(' ', '') != gold.target.replace(' ', ''):
                    with open(os.path.join(patch_lines_dir, id + ".txt"), 'w') as f:
                        f.write(ref + '\n')
                    f.close()
                index += 1
            logger.info("***** End building valid dataset*****")
        if change:
            model = torch.nn.DataParallel(model)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    log_file.write("Finish and take {}\n".format(get_elapse_time(t0)))
    log_file.close()


if __name__ == '__main__':
    main()

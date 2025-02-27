import logging
import os
import re
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PairFeatures(object):
    def __init__(self,
                 idx,
                 input_ids,
                 input_mask):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 buggy_method,
                 buggy_lines,
                 fix_lines
                 ):
        self.idx = idx
        self.buggy_method = buggy_method
        self.buggy_lines = buggy_lines
        self.fix_lines = fix_lines


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 gen_src_ids,
                 gen_src_mask,
                 gen_tgt_ids,
                 gen_tgt_mask,
                 red_buggy_ids,
                 red_buggy_mask
                 ):
        self.example_id = example_id
        self.gen_src_ids = gen_src_ids
        self.gen_tgt_ids = gen_tgt_ids
        self.gen_src_mask = gen_src_mask
        self.gen_tgt_mask = gen_tgt_mask
        self.red_buggy_ids = red_buggy_ids
        self.red_buggy_mask = red_buggy_mask


def readLines(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
        f.close()
    return lines


def prepare_examples_dir(dir, mode):
    if mode == "train":
        ids_f = os.path.join(dir, "trn.ids")
    else:
        ids_f = os.path.join(dir, "valid.ids")
    buggy_methods_dir = os.path.join(dir, "buggy_methods")
    buggy_lines_dir = os.path.join(dir, "buggy_lines")
    fix_lines_dir = os.path.join(dir, "fix_lines")
    return prepare_CR3_examples(ids_f, buggy_methods_dir, buggy_lines_dir, fix_lines_dir)


def prepare_CR3_examples(ids_f, buggy_methods_dir, buggy_lines_dir, fix_lines_dir):
    ids = readLines(ids_f)
    examples = []
    idx = 0
    for id in tqdm(ids):
        buggy_line = open(os.path.join(buggy_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        fix_line = open(os.path.join(fix_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        buggy_method = readLines(os.path.join(buggy_methods_dir, id + ".txt"))
        buggy_method = '\n'.join(buggy_method)
        examples.append(Example(
            idx=idx,
            buggy_method=buggy_method.strip(),
            buggy_lines=buggy_line,
            fix_lines=fix_line
        ))
        idx += 1
        # print(idx,input,output)
    return examples


def convert_examples_to_features(examples, gen_tokenizer, red_tokenizer, args, stage=None):
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        source_ids, source_mask, target_ids, target_mask, red_buggy_ids, red_buggy_mask = None, None, None, None, None, None
        # 生成器数据处理
        if args.generator=='CodeBERT':
            # source
            source_tokens = gen_tokenizer.tokenize(example.buggy_lines)[:args.max_source_length - 2]
            source_tokens = [gen_tokenizer.cls_token] + source_tokens + [gen_tokenizer.sep_token]
            source_ids = gen_tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = args.max_source_length - len(source_ids)
            source_ids += [gen_tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length

            # target
            target_tokens = gen_tokenizer.tokenize(example.fix_lines)[:args.max_target_length - 2]
            target_tokens = [gen_tokenizer.cls_token] + target_tokens + [gen_tokenizer.sep_token]
            target_ids = gen_tokenizer.convert_tokens_to_ids(target_tokens)
            target_mask = [1] * len(target_ids)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [gen_tokenizer.pad_token_id] * padding_length
            target_mask += [0] * padding_length
        else:
            buggy_method = example.buggy_method.split('\n')
            for ind in range(len(buggy_method)):
                if example.buggy_lines in buggy_method[ind]:
                    buggy_method[ind] = " <BUGS> " + example.buggy_lines + " <BUGE> "
            input = '\n'.join(buggy_method)
            input = re.sub('\s+', ' ', input)
            output = re.sub('\s+', ' ', " <FIXS> " + example.fix_lines.strip() + " <FIXE> ")
            if args.generator=='CodeT5':
                # source
                source_str = input.replace('</s>', '<unk>')
                source_ids = gen_tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
                assert source_ids.count(gen_tokenizer.eos_token_id) == 1
                
                # targat
                target_str = output
                target_str = target_str.replace('</s>', '<unk>')
                target_ids = gen_tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length', truncation=True)
                assert target_ids.count(gen_tokenizer.eos_token_id) == 1
            else:
                # source
                source_tokens = gen_tokenizer.tokenize(input)[:args.max_source_length - 4]
                source_tokens = [gen_tokenizer.cls_token, "<encoder-decoder>", gen_tokenizer.sep_token] + source_tokens + [gen_tokenizer.sep_token]
                source_ids = gen_tokenizer.convert_tokens_to_ids(source_tokens)
                source_mask = [1] * (len(source_tokens))
                padding_length = args.max_source_length - len(source_ids)
                source_ids += [gen_tokenizer.pad_token_id] * padding_length
                source_mask += [0] * padding_length

                # target
                target_tokens = gen_tokenizer.tokenize(output)[:args.max_target_length - 2]
                target_tokens = [gen_tokenizer.cls_token] + target_tokens + [gen_tokenizer.sep_token]
                target_ids = gen_tokenizer.convert_tokens_to_ids(target_tokens)
                target_mask = [1] * len(target_ids)
                padding_length = args.max_target_length - len(target_ids)
                target_ids += [gen_tokenizer.pad_token_id] * padding_length
                target_mask += [0] * padding_length
        # 奖励模型数据处理
        if args.reward == 'APPT':
            encoding = red_tokenizer(example.buggy_method)
            if len(encoding['input_ids']) > args.reward_max_length:
                half_length = int(args.reward_max_length / 2)
                red_buggy_ids = encoding['input_ids'][:half_length] + encoding['input_ids'][-half_length:]
                #        encoding['token_type_ids'] = encoding['token_type_ids'][:half_length] + encoding['token_type_ids'][-half_length:]
                red_buggy_mask = encoding['attention_mask'][:half_length] + encoding['attention_mask'][-half_length:]
                # encoding.pop('token_type_ids')
            else:
                red_buggy_ids = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), args.reward_max_length)]
                red_buggy_mask = encoding['attention_mask'] + [0 for i in range(len(encoding['attention_mask']), args.reward_max_length)]
                # encoding.pop('token_type_ids')
        else:
            pass

        # if example_index < 5 and stage == 'train':
        #     logger.info("*** Example ***")
        #     logger.info("idx: {}".format(example.idx))

        #     logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
        #     logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
        #     logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

        #     logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
        #     logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
        #     logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_id=example_index,
                gen_src_ids=source_ids,
                gen_src_mask=source_mask,
                gen_tgt_ids=target_ids,
                gen_tgt_mask=target_mask,
                red_buggy_ids=red_buggy_ids,
                red_buggy_mask=red_buggy_mask
            )
        )
    return features


def load_gen_data(args, config, filename, mode="train", number=None):
    examples = prepare_examples_dir(filename, mode)
    features = convert_examples_to_features(examples, config.generator[2], config.reward[2], args, stage=mode)
    all_source_ids, all_source_mask, all_target_ids, all_target_mask, buggy_ids, buggy_mask = None, None, None, None, None, None
    if number:
        examples = examples[:number]
        features = features[:number]
    all_source_ids = torch.tensor([f.gen_src_ids for f in features], dtype=torch.long)
    if args.generator=='CodeT5':
        all_source_mask = all_source_ids.ne(config.generator[2].pad_token_id)
    else:
        all_source_mask = torch.tensor([f.gen_src_mask for f in features], dtype=torch.long)
    if features[0].gen_tgt_ids:
        all_target_ids = torch.tensor([f.gen_tgt_ids for f in features], dtype=torch.long)
        if args.generator=='CodeT5':
            all_target_mask = all_target_ids.ne(config.generator[2].pad_token_id)
        else:
            all_target_mask = torch.tensor([f.gen_tgt_mask for f in features], dtype=torch.long)
    if args.reward != "Metric":
        buggy_ids = torch.tensor([f.red_buggy_ids for f in features], dtype=torch.long)
        buggy_mask = torch.tensor([f.red_buggy_mask for f in features], dtype=torch.long)
    else:
        buggy_ids = all_source_ids
        buggy_mask = all_source_mask
    data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, buggy_ids, buggy_mask)
    return examples, data


def patch_text2data(args, texts, tokenizer):
    features = []
    for idx, text in enumerate(texts):
        encoding = tokenizer(text, padding=True)
        if len(encoding['input_ids']) > args.reward_max_length:
            half_length = int(args.reward_max_length / 2)
            encoding['input_ids'] = encoding['input_ids'][:half_length] + encoding['input_ids'][-half_length:]
            encoding['attention_mask'] = encoding['attention_mask'][:half_length] + encoding['attention_mask'][-half_length:]
        else:
            encoding['input_ids'] = encoding['input_ids'] + [0 for i in range(len(encoding['input_ids']), args.reward_max_length)]
            encoding['attention_mask'] = encoding['attention_mask'] + [0 for i in range(len(encoding['attention_mask']), args.reward_max_length)]
        features.append(
            PairFeatures(
                idx=idx,
                input_ids=encoding['input_ids'],
                input_mask=encoding['attention_mask']
            )
        )
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    return input_ids.to(args.device), input_mask.to(args.device)
    
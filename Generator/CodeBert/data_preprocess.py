import logging
import os
import re

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def readLines(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            lines.append(line.strip())
        f.close()
    return lines


def prepare_examples_dir(dir, mode="train"):
    if mode == "train":
        ids_f = os.path.join(dir, "trn.ids")
    else:
        ids_f = os.path.join(dir, "valid.ids")
    buggy_lines_dir = os.path.join(dir, "buggy_lines")
    fix_lines_dir = os.path.join(dir, "fix_lines")
    return prepare_CR3_examples(ids_f, buggy_lines_dir, fix_lines_dir)


def prepare_CR3_examples(ids_f, buggy_lines_dir, fix_lines_dir):
    ids = readLines(ids_f)
    examples = []
    idx = 0
    for id in tqdm(ids):
        buggy_line = open(os.path.join(buggy_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        fix_line = open(os.path.join(fix_lines_dir, id + ".txt"), 'r', encoding='utf8').read().strip()
        examples.append(Example(
            idx=idx,
            source=buggy_line,
            target=fix_line,
        ))
        idx += 1
        # print(idx,input,output)
    return examples


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def load_gen_data(args, tokenizer, filename, mode="train"):
    examples = prepare_examples_dir(filename, mode)
    features = convert_examples_to_features(examples, tokenizer, args, stage=mode)
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    return examples, data


def read_test_examples(filename):
    examples = []
    idx = 0
    with open(filename) as f1:
        for line1 in f1:
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target="",
                )
            )
            idx += 1
    return examples


def load_gen_data_test(args, tokenizer, filename, target="train"):
    examples = prepare_examples_dir(filename, target)
    features = convert_examples_to_features(examples, tokenizer, args, stage="test")
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
    data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    return examples, data

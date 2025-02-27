import torch
from Generator.CodeBert.model import build_or_load_gen_model as codebert
from Generator.CodeT5.model import build_or_load_gen_model as codet5
from Generator.UniXcoder.model import build_or_load_gen_model as unixcoder
from Reward.APPT.Model import build_or_load_dis_model as appt

def load_CodeBert(args):
    config, model, tokenizer = codebert(args)
    return (config, model, tokenizer)

def load_CodeT5(args):
    config, model, tokenizer = codet5(args)
    return (config, model, tokenizer)

def load_UniXcoder(args):
    config, model, tokenizer= unixcoder(args)
    return (config, model, tokenizer)

def load_APPT(args):
    config, model, tokenizer = appt(args)
    return (config, model, tokenizer)

def load_Patcherizer(args):
    return (None, None, None)

def load_Metric(args):
    return (None, None, None)

generators = {'CodeBERT':load_CodeBert, 'CodeT5':load_CodeT5, 'UniXcoder':load_UniXcoder}
rewards = {'APPT':load_APPT, 'Patcherizer':load_Patcherizer, 'Metric':load_Metric}


class Config(object):
    '''
    配置参数
    '''
    def __init__(self, args):
        # 加载生成器模型
        self.generator = generators[args.generator](args)
        
        # 加载奖励模型
        self.reward = rewards[args.reward](args)
    
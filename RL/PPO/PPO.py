import logging
import torch
import torch.optim as optim
import torch.nn as nn

logger = logging.getLogger(__name__)

# PPO 网络结构
class PPO(nn.Module):
    def __init__(self, generator_model, generator_tokenizer, patch_evaluator):
        super(PPO, self).__init__()
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.lsm = nn.LogSoftmax(dim=-1)
        self.patch_evaluator = patch_evaluator
        self.old_log_probs = None
        if self.patch_evaluator:
            for param in self.patch_evaluator.parameters():
                param.requires_grad = False

    def forward(self, source_ids, source_mask, target_ids, target_mask, generator):
        # logger.info("******")
        # 生成补丁
        if generator=='CodeBERT':
            # logger.info("*")
            _, _, _, logits = self.generator_model(source_ids, source_mask, target_ids, target_mask)
            # logger.info('logits')
            # logger.info(logits.shape)
            with torch.no_grad():
                preds = self.generator_model(source_ids=source_ids, source_mask=source_mask)
            # logger.info('preds')
            # logger.info(preds.shape)
        elif generator=='CodeT5':
            # print("**")
            logits = self.generator_model(source_ids, source_mask, target_ids, target_mask).logits
            # print(logits.shape)
            with torch.no_grad():
                preds = self.generator_model.generate(source_ids,
                                                      attention_mask=source_ids.ne(self.generator_tokenizer.pad_token_id),
                                                      use_cache=True,
                                                      num_beams=5,
                                                      early_stopping=True,
                                                      max_length=256,
                                                      pad_token_id=self.generator_tokenizer.pad_token_id)
                
                # print(preds.shape)
                preds = torch.nn.functional.pad(preds, (0, 256-preds.shape[1]), value=self.generator_tokenizer.pad_token_id)
            masked_logits = self.lsm(logits)
            # logger.info('masked_logits1')
            # logger.info(target_mask.dtype)
            # logger.info(masked_logits.shape)
            # logger.info(masked_logits.dtype)        
            masked_logits[~target_mask.unsqueeze(-1).expand_as(masked_logits)] = torch.tensor(float('-inf'), dtype=masked_logits.dtype)
            # logger.info('masked_logits2')
            # logger.info(masked_logits.shape)
            # logger.info(masked_logits.dtype)  
            patch_out = torch.argmax(masked_logits, dim=-1)
            # logger.info('patch_out')
            # logger.info(patch_out.shape)
        else:
            # print("***")
            _, _, _, logits = self.generator_model(source_ids, target_ids)
            with torch.no_grad():
                preds = self.generator_model(source_ids)
            # print(logits.shape)

        masked_logits = self.lsm(logits)
        # logger.info('masked_logits1')
        # logger.info(target_mask.dtype)
        # logger.info(masked_logits.shape)
        # logger.info(masked_logits.dtype)
        patch_out = torch.argmax(masked_logits, dim=-1)     
        patch_out = patch_out * target_mask.long()
        # logger.info('patch_out')
        # logger.info(patch_out.shape)
        return logits, patch_out, preds

    def evaluate(self, buggy_ids, buggy_mask, patch_ids, patch_mask):
        # 评估生成的补丁
        reward = torch.sigmoid(self.patch_evaluator(buggy_ids, buggy_mask, patch_ids, patch_mask))
        return reward


# 奖励平滑
class RewardBuffer:
    def __init__(self, config):
        self.alpha = config.reward_smooth_alpha
        self.last_reward = None
    
    def smooth(self, reward):
        if self.last_reward is None:
            self.last_reward = reward
        else:
            self.last_reward = self.alpha * self.last_reward + (1 - self.alpha) * reward
        return self.last_reward
from codebleu import calc_codebleu
import code_bert_score

# Test
# prediction1 = "public int add(int a, int b) {\n    return a + b;\n}"
# reference1 = "public int add(int a, int b) {\n return a + b;\n}"
# prediction2 = "public int add(int a, int b) {\n    return a + b;\n}"
# reference2 = "public int sum(int first, int second) {\n return first + second;\n}"

# result = calc_codebleu([reference1, reference2], [prediction1, prediction2], lang='java', weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
# print(result)

# ans = code_bert_score.score(cands=[prediction1, prediction2], refs=[reference1, reference2], lang='java')
# print(ans)

def get_score(buggy, reference, prediction, model_type):
    result = calc_codebleu(references=[reference], predictions=[prediction], lang='java')
    if result["dataflow_match_score"] != 0:
        codebleu = result["codebleu"]
    else:
        codebleu = result["ngram_match_score"]*0.3 + result["weighted_ngram_match_score"]*0.3 + result["syntax_match_score"]*0.4
    bert_score_rp = code_bert_score.score(cands=[prediction], refs=[reference], lang='java', model_type=model_type)[2].item()
    bert_score_bp = code_bert_score.score(cands=[prediction], refs=[buggy], lang='java', model_type=model_type)[2].item()
    return codebleu+bert_score_rp-bert_score_bp
"""Pinned LM Evaluation Harness adapter; protocol assumptions are in the spec."""
import argparse
import json
from pathlib import Path

import tiktoken
import torch

from .evaluate import load_model

TASKS = {"arc_easy": "acc_norm,none", "boolq": "acc,none", "hellaswag": "acc_norm,none",
         "lambada_openai": "acc,none", "openbookqa": "acc_norm,none", "piqa": "acc_norm,none",
         "social_iqa": "acc,none", "winogrande": "acc,none"}
HARNESS_COMMIT = "ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9"


@torch.no_grad()
def score_tokens(model, context, continuation, device):
    if not continuation or len(continuation) > model.cfg.seq_len:
        raise ValueError("continuation must contain 1..context_length tokens")
    full = (context + continuation)[-(model.cfg.seq_len + 1):]
    if len(full) <= len(continuation):
        raise ValueError("need a context token before continuation")
    inputs = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    logits = model(inputs)[0, -len(continuation):].float().log_softmax(-1)
    target = torch.tensor(continuation, device=device)
    return float(logits.gather(1, target[:,None]).sum()), bool((logits.argmax(-1) == target).all())


def make_adapter(checkpoint, device):
    # Optional dependency is imported only for downstream runs, not training.
    from lm_eval.api.model import TemplateLM
    from lm_eval import utils

    class XSAHarness(TemplateLM):
        def __init__(self):
            super().__init__()
            self.model, self.state = load_model(checkpoint, device)
            self.encoding = tiktoken.get_encoding("gpt2")

        @property
        def eot_token_id(self):
            return self.encoding.eot_token

        @property
        def tokenizer_name(self):
            return "gpt2-tiktoken-0.9.0"

        def tok_encode(self, string, add_special_tokens=None, **kwargs):
            return self.encoding.encode(string, allowed_special={"<|endoftext|>"})

        def _loglikelihood_tokens(self, requests, **kwargs):
            results = []
            for key, context, continuation in requests:
                result = score_tokens(self.model, context or [self.eot_token_id], continuation, device)
                if key is not None:
                    self.cache_hook.add_partial("loglikelihood", key, result)
                results.append(result)
            return results

        def loglikelihood_rolling(self, requests, **kwargs):
            result = []
            for req in requests:
                windows = utils.get_rolling_token_windows(token_list=self.tok_encode(req.args[0]), prefix_token=self.eot_token_id,
                                                         max_seq_len=self.model.cfg.seq_len, context_len=1)
                pairs = [utils.make_disjoint_window(w) for w in windows]
                result.append(sum(score_tokens(self.model, c, t, device)[0] for c,t in pairs))
            return result

        def generate_until(self, requests, **kwargs):
            raise NotImplementedError("the eight selected tasks use loglikelihood; generation is outside this protocol")

    return XSAHarness()


def main():
    from lm_eval import evaluator
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--num-fewshot",type=int,default=0);p.add_argument("--limit",type=int)
    a=p.parse_args()
    result=evaluator.simple_evaluate(model=make_adapter(a.checkpoint,a.device),tasks=list(TASKS),num_fewshot=a.num_fewshot,
                                    limit=a.limit,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234,
                                    apply_chat_template=False,log_samples=True)
    result["reproduction_protocol"]={"harness_commit":HARNESS_COMMIT,"selected_metrics":TASKS,"fewshot":a.num_fewshot,
                                     "assumption":"paper does not specify prompting/few-shot settings or LAMBADA variant; use pinned harness defaults and zero-shot",
                                     "exact_table2_replication":False}
    Path(a.output).write_text(json.dumps(result,indent=2,default=str)+'\n')


if __name__=="__main__":
    main()

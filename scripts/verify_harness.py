"""Exercise the actual pinned harness interfaces without downloading task corpora."""
import argparse
import importlib.metadata
import json
from types import SimpleNamespace

from xsa.downstream import make_adapter, score_tokens, TASKS, HARNESS_COMMIT

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
model=make_adapter(a.checkpoint,'cpu')
req=SimpleNamespace(args=('The careful researcher',' checks the result.'))
score=model.loglikelihood([req])[0]
c,t=model._encode_pair(*req.args)
assert score==score_tokens(model.model,c,t,'cpu')
rolling=model.loglikelihood_rolling([SimpleNamespace(args=('A short test.',))])
assert len(rolling)==1 and rolling[0]<0
from lm_eval.tasks import TaskManager
manager=TaskManager()
missing=set(TASKS)-set(manager.all_tasks)
assert not missing, missing
print(json.dumps({'passed':True,'harness_version':importlib.metadata.version('lm_eval'),'harness_commit':HARNESS_COMMIT,
                  'likelihood':score,'rolling':rolling,'task_ids_verified':list(TASKS),'full_downstream_tasks_run':False},indent=2))

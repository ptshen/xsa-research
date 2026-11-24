"""Reproduce the compact-artifact audit and late-training figure; no training."""
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xsa.direction_a import screen_summary, source_hashes

root = Path('evidence/direction_a')
out = root / 'results'
names = ['pilot-sa-s17', 'pilot-xsa-s17', 'direction-a-tau025-s17',
         'direction-a-tau050-s17', 'direction-a-tau075-s17']
labels = ['SA', 'XSA', 'A τ=0.25', 'A τ=0.50', 'A τ=0.75']
records, train_rows, manifests, diagnostics, events = [], [], [], [], []
for name in names:
    path = out / name
    record = json.loads((path / 'evaluation.json').read_text())
    rows = [json.loads(s) for s in (path / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in rows if r['type'] == 'train']
    assert [r['update'] for r in train] == list(range(1, 2049)), name
    assert all(r['tokens'] == r['update'] * 32768 for r in train), name
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in train), name
    assert record['tokens'] == 67108864 and record['update'] == 2048, name
    assert len(record['window_losses']) == 64 and record['evaluated_tokens'] == 32768, name
    mean = sum(record['window_losses']) / 64
    assert abs(mean - record['cross_entropy']) < 1e-12, name
    assert abs(math.exp(mean) - record['perplexity']) < 1e-10, name
    files = list(path.glob('manifest-*.json')); assert len(files) == 1, name
    manifest = json.loads(files[0].read_text())
    assert manifest['software'] == record['software'], name
    d = json.loads((path / 'diagnostics.json').read_text())
    assert (d['samples'], d['context'], d['seed'], d['split'], d['checkpoint_tokens']) == (4,128,7017,'train',67108864), name
    assert len(d['layers']) == 6, name
    for layer in d['layers']:
        required = ['alignment_before','alignment_after','aggregate_norm','corrected_norm',
                    'removed_update_norm','zero_output_fraction','near_zero_output_fraction']
        if record['attention'] == 'alignment_cap': required += ['cap_activation_fraction']
        for key in required:
            assert len(layer[key]) == 6 and all(math.isfinite(x) for x in layer[key]), (name,key)
            if key.endswith('fraction'): assert all(0 <= x <= 1 for x in layer[key]), (name,key)
            if key.endswith('norm'): assert all(x >= 0 for x in layer[key]), (name,key)
    if record['attention'] == 'alignment_cap':
        assert all(r['alignment_tau'] == record['alignment_tau'] for r in rows), name
        assert d['alignment_tau'] == record['alignment_tau'], name
    records.append(record); train_rows.append(train); manifests.append(manifest); diagnostics.append(d); events.append(rows)

base_config = None
for record, train, manifest in zip(records, train_rows, manifests):
    assert [r['batch_sha256'] for r in train] == [r['batch_sha256'] for r in train_rows[0]]
    assert record['software'] == records[0]['software']
    assert manifest['parameter_counts'] == manifests[0]['parameter_counts']
    cfg = copy.deepcopy(manifest['config'])
    cfg.pop('label', None); cfg['model'].pop('attention_type'); cfg['model'].pop('alignment_tau', None)
    if base_config is None: base_config = cfg
    assert cfg == base_config
assert manifests[0]['parameter_counts']['total_unique'] == 29938944
assert screen_summary(records) == json.loads((out / 'summary.json').read_text())
identity = json.loads((out / 'identity.json').read_text())
assert len(identity) == 5
for item, record in zip(identity, records):
    assert item['checkpoint_sha256'] == record['checkpoint_sha256']
    assert item['all_batch_hashes_match_baseline'] and item['final_data_state_matches_baseline']
proof = json.loads((root / 'verification.json').read_text())
assert proof['source_sha256'] == source_hashes()
suites = list(ET.parse(root / 'tests.xml').getroot().iter('testsuite'))
assert sum(int(s.attrib['tests']) for s in suites) == 49
assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors','failures','skipped'))
assert hashlib.sha256(Path('evidence/pilot/data_manifest.json').read_bytes()).hexdigest() == records[0]['data_manifest_sha256']
ledger = json.loads((root / 'job_ledger.json').read_text())
assert all(j['state'] == 'completed' and j['exit_code'] == '0:0' for j in ledger)
gpu = [j for j in ledger if j['kind'] in ('verify','train')]
assert sum(j['actual_elapsed_seconds'] for j in gpu) == 2049 <= 7200
assert sum(j['seconds'] for j in gpu) == 5580 <= 7200
assert sum(j['actual_elapsed_seconds'] for j in ledger if j['kind'] == 'collect') == 36 <= 900
assert sum(j['seconds'] for j in ledger if j['kind'] == 'collect') == 300 <= 900
ordered = sorted(gpu, key=lambda j:j['start'])
assert all(a['end'] <= b['start'] for a,b in zip(ordered, ordered[1:]))
original = json.loads(Path('evidence/job_ledger.json').read_text())['jobs']
assert sum(j['actual_elapsed_seconds'] for j in original if j['kind'] == 'smoke') == 49
assert sum(j['reserved_seconds'] for j in original if j['kind'] == 'smoke') == 480
manifest = json.loads(Path('DIRECTION_A_MANIFEST.json').read_text())
assert manifest['project_bytes'] < 8 * 1024**3
assert manifest['all_stage_jobs_terminal'] and manifest['sequential_gpu_allocations_verified']
assert manifest['summary'] == screen_summary(records)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for label, rows in zip(labels, events):
    late = [r for r in rows if r['type'] == 'eval' and r['update'] >= 1024]
    for ax, key in zip(axes, ['tokens', 'elapsed_seconds']):
        ax.plot([r[key] for r in late], [r['cross_entropy'] for r in late], marker='o', label=label)
        ax.set_ylabel('Validation cross-entropy'); ax.grid(alpha=.25); ax.legend()
axes[0].set_xlabel('Training tokens'); axes[1].set_xlabel('Elapsed training/evaluation seconds')
fig.suptitle('Second half of training — validation screen, seed 17')
fig.savefig(out / 'curves_zoom.png', dpi=160); plt.close(fig)
audit = {'passed':True,'updates_per_method':2048,'tokens_per_method':67108864,'methods':5,
         'all_batch_hashes_identical':True,'configs_match_except_method_threshold_label':True,
         'software_and_parameter_counts_identical':True,'finite_training_metrics':True,
         'means_perplexities_and_summary_reproduced':True,'diagnostic_shapes_and_values_checked':True,
         'numerical_source_matches_cuda_gate':True,'cuda_tests_passed':49,'cuda_tests_skipped':0,
         'checkpoint_identity_source':'successful CPU collector 14074923; actual checkpoint loading and checksums',
         'gpu_seconds':2049,'cpu_collection_seconds':36,'sequential_allocations_verified':True,
         'original_smoke_seconds_unchanged':49,'project_bytes_at_collection':manifest['project_bytes']}
(root / 'final_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))

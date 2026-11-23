import copy
import math

import pytest
import torch

from xsa.model import GPT, ModelConfig, attention, cap_alignment, project
from xsa.evaluate import comparison_hash, load_model
from xsa.initialize import architecture, create_initial_state
from xsa.train import Trainer
from test_pipeline import config, tiny_data


def reference(y, v, tau, eps=1e-12):
    u = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
    c = (y * u).sum(-1, keepdim=True)
    r = y - c * u
    b = tau / math.sqrt(1 - tau * tau) * r.norm(dim=-1, keepdim=True)
    return r + torch.maximum(torch.minimum(c, b), -b) * u


def test_tau_zero_matches_xsa_forward_and_backward():
    torch.manual_seed(13)
    y, v = [torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    a, b = cap_alignment(y, v, 0), project(y, v)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    for x, z in zip(torch.autograd.grad(a.square().sum(), (y, v)),
                    torch.autograd.grad(b.square().sum(), (y, v))):
        torch.testing.assert_close(x, z, atol=0, rtol=0)


@pytest.mark.parametrize('tau', [.25, .5, .75])
def test_cap_geometry_reference_and_gradients(tau):
    torch.manual_seed(18)
    v = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
    y = (torch.randn_like(v) + 3 * v.detach()).requires_grad_()
    z = cap_alignment(y, v, tau)
    expected = reference(y, v, tau)
    torch.testing.assert_close(z, expected, atol=2e-14, rtol=2e-13)
    cosine = torch.nn.functional.cosine_similarity(z, v, dim=-1)
    assert cosine.abs().max() <= tau + 1e-12
    for a, b in zip(torch.autograd.grad(z.square().sum(), (y, v), retain_graph=True),
                    torch.autograd.grad(expected.square().sum(), (y, v))):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda y, v: cap_alignment(y, v, tau), (y, v))


def test_inactive_signed_and_degenerate_cases():
    v = torch.tensor([[1., 0.]], dtype=torch.float64)
    for y in (torch.tensor([[.1, 2.]], dtype=torch.float64), torch.tensor([[0., 2.]], dtype=torch.float64)):
        torch.testing.assert_close(cap_alignment(y, v, .5), y, atol=0, rtol=0)
    for sign in (-1, 1):
        y = torch.tensor([[sign * 4., 2.]], dtype=torch.float64)
        z = cap_alignment(y, v, .5)
        torch.testing.assert_close(z, torch.tensor([[sign * 2 / math.sqrt(3), 2.]], dtype=torch.float64))
        torch.testing.assert_close(cap_alignment(sign * v, v, .999), torch.zeros_like(v))
    y = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
    for vv in (torch.zeros_like(y), v * 1e-14, v * 1e-10, v):
        vv = vv.requires_grad_()
        z = cap_alignment(y, vv, .5)
        assert torch.isfinite(z).all()
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(z.sum(), (y, vv), retain_graph=True))
    torch.testing.assert_close(cap_alignment(y, torch.zeros_like(y), .5), y)
    # Below epsilon the baseline uses a clamped, non-unit normalization.
    torch.testing.assert_close(cap_alignment(y, v * 1e-14, 0), project(y, v * 1e-14))


@pytest.mark.parametrize('tau', [-.1, 1., 1.1, float('nan'), float('inf')])
def test_invalid_thresholds(tau):
    with pytest.raises(ValueError, match='tau'):
        ModelConfig(attention_type='alignment_cap', alignment_tau=tau)
    with pytest.raises(ValueError, match='tau'):
        cap_alignment(torch.ones(2), torch.ones(2), tau)


def test_reference_sdpa_first_position_and_causality():
    torch.manual_seed(2)
    q, k, v = [torch.randn(1, 2, 5, 8, requires_grad=True) for _ in range(3)]
    ref = attention(q, k, v, 'alignment_cap', backend='reference', tau=.5)
    fast = attention(q, k, v, 'alignment_cap', tau=.5)
    torch.testing.assert_close(fast, ref, atol=1e-6, rtol=2e-5)
    for a, b in zip(torch.autograd.grad(ref.square().sum(), (q, k, v), retain_graph=True),
                    torch.autograd.grad(fast.square().sum(), (q, k, v))):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(fast[:, :, 0], torch.zeros_like(v[:, :, 0]), atol=1e-6, rtol=0)
    changed = [a.detach().clone() for a in (q, k, v)]
    for a in changed: a[:, :, 3:] += 20
    torch.testing.assert_close(attention(*changed, 'alignment_cap', tau=.5)[:, :, :3], fast[:, :, :3])


def test_parameters_shared_initialization_legacy_checkpoint_and_hashes(tmp_path):
    c = config(); data = tiny_data(tmp_path / 'data')
    initial = tmp_path / 'initial.pt'; create_initial_state(c, initial)
    c['initial_state_path'] = str(initial)
    baseline = Trainer(c, data, tmp_path / 'sa')
    other = copy.deepcopy(c); other['model'].update(attention_type='alignment_cap', alignment_tau=.5)
    capped = Trainer(other, data, tmp_path / 'cap')
    assert architecture(c) == architecture(other)
    assert baseline.initial_hash == capped.initial_hash
    assert baseline.model.parameter_counts() == capped.model.parameter_counts()
    assert comparison_hash(c) == comparison_hash(other)
    changed = copy.deepcopy(other); changed['train']['learning_rate'] *= 2
    assert comparison_hash(c) != comparison_hash(changed)
    restored, _ = load_model(tmp_path / 'sa' / 'checkpoint.pt', 'cpu')
    x = torch.randint(20, (1, 8)); baseline.model.eval()
    torch.testing.assert_close(restored(x), baseline.model(x), atol=0, rtol=0)
    for a, b in zip(baseline.model.parameters(), capped.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cap_resume_and_reject_changed_tau(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('requires allocated CUDA GPU')
    c = config(); c['model'].update(attention_type='alignment_cap', alignment_tau=.5)
    if device == 'cuda': c['train']['precision'] = 'bfloat16'
    data = tiny_data(tmp_path / 'data')
    full = Trainer(c, data, tmp_path / 'full', device); full.run()
    split = Trainer(c, data, tmp_path / 'split', device); split.run(stop_after=2)
    resumed = Trainer(c, data, tmp_path / 'split', device, resume=True); resumed.run()
    assert full.batch_hashes == resumed.batch_hashes
    assert full.stream.state_dict() == resumed.stream.state_dict()
    for k, v in full.model.state_dict().items():
        torch.testing.assert_close(v, resumed.model.state_dict()[k], atol=0, rtol=0)
    for pid, state in full.optimizer.state_dict()['state'].items():
        for k, v in state.items():
            torch.testing.assert_close(v, resumed.optimizer.state_dict()['state'][pid][k], atol=0, rtol=0)
    changed = copy.deepcopy(c); changed['model']['alignment_tau'] = .75
    with pytest.raises(ValueError, match='configuration'):
        Trainer(changed, data, tmp_path / 'split', device, resume=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_cap_precision(dtype):
    if not torch.cuda.is_available(): pytest.skip('requires allocated CUDA GPU')
    torch.manual_seed(9)
    v = torch.randn(2, 6, 32, 64, device='cuda', dtype=dtype)
    y = (torch.randn_like(v) + 3 * v).requires_grad_(); v.requires_grad_()
    z = cap_alignment(y, v, .5)
    fp32 = cap_alignment(y.float(), v.float(), .5)
    torch.testing.assert_close(z, fp32.to(dtype), atol=0, rtol=0)
    cosine = torch.nn.functional.cosine_similarity(z.float(), v.float(), dim=-1).abs()
    assert cosine.max() <= .5 + (2e-6 if dtype == torch.float32 else .002)
    z.float().square().mean().backward()
    assert torch.isfinite(y.grad).all() and torch.isfinite(v.grad).all()


def test_baseline_outputs_and_gradients_unchanged():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('frozen_baseline', 'evidence/direction_a/baseline_model.py')
    old = importlib.util.module_from_spec(spec); sys.modules[spec.name] = old; spec.loader.exec_module(old)
    for kind in ('sa', 'xsa'):
        kwargs = dict(vocab_size=31, n_layers=2, d_model=12, n_heads=2, head_dim=8, seq_len=8, attention_type=kind)
        torch.manual_seed(17); before = old.GPT(old.ModelConfig(**kwargs))
        torch.manual_seed(17); after = GPT(ModelConfig(**kwargs))
        x = torch.randint(31, (2, 8))
        a, b = before(x), after(x)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for p, q in zip(before.parameters(), after.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


def test_cap_diagnostic_activation_and_zero_outputs():
    from xsa.diagnostics import diagnostic_stats
    v = torch.tensor([[[[1., 0.], [1., 0.], [1., 0.], [0., 0.]]]])
    y = torch.tensor([[[[1., 0.], [0., 1.], [4., 2.], [0., 1.]]]])
    z = cap_alignment(y, v, .5)
    stats = diagnostic_stats(v, v, v, y, z, tau=.5)
    assert stats['cap_activation_fraction'] == [.5]
    assert stats['zero_output_fraction'] == [.25]
    assert stats['near_zero_output_fraction'] == [.25]


def test_threshold_recorded_in_metrics_and_evaluation(tmp_path):
    import json
    from xsa.evaluate import evaluate
    c = config(); c['model'].update(attention_type='alignment_cap', alignment_tau=.25)
    data = tiny_data(tmp_path / 'data')
    trainer = Trainer(c, data, tmp_path / 'run'); trainer.run(stop_after=1)
    rows = [json.loads(s) for s in (tmp_path / 'run' / 'metrics.jsonl').read_text().splitlines()]
    assert all(r['alignment_tau'] == .25 and r['attention'] == 'alignment_cap' for r in rows)
    assert evaluate(tmp_path / 'run' / 'checkpoint.pt', data)['alignment_tau'] == .25

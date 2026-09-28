from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_vanilla_warmup_uses_the_first_real_prompt_and_short_budget(monkeypatch) -> None:
    from argparse import Namespace

    import torch

    import Benchmark.common.vanilla_inference as vanilla

    input_ids = torch.tensor([[1, 2, 3, 4]])
    calls = []

    def fake_generate(model, observed_ids, args):
        calls.append((model, observed_ids.clone(), args.max_new_tokens))

    monkeypatch.setattr(vanilla, "_generate", fake_generate)

    vanilla._warmup_sample(
        "model",
        input_ids,
        Namespace(warmup_runs=2, max_new_tokens=2048),
        torch.device("cpu"),
    )

    assert len(calls) == 2
    assert all(model == "model" for model, _, _ in calls)
    assert all(torch.equal(observed, input_ids) for _, observed, _ in calls)
    assert all(max_new_tokens == 8 for _, _, max_new_tokens in calls)

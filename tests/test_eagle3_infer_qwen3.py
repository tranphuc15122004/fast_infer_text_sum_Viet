from types import SimpleNamespace

import torch

from Benchmark.eagle3_infer_qwen3 import timed_generate


class _FakeEagleModel:
    tokenizer = SimpleNamespace(eos_token_id=1)

    def eagenerate(self, input_ids, **kwargs):
        return (
            torch.tensor([[10, 11, 20, 99, 22]]),
            3,
            2,
            0.2,
            [2, 2],
            {"draft_tokens_proposed": 8},
        )


def test_timed_generate_truncates_at_supplied_qwen_stop_tokens():
    result = timed_generate(
        _FakeEagleModel(),
        torch.tensor([[10, 11]]),
        temperature=0.0,
        max_new_tokens=8,
        total_token=18,
        spec=True,
        is_llama3=False,
        include_phase_timings=True,
        stop_token_ids=[99],
    )

    output_ids, output_tokens, _steps, _elapsed, acceptance, phases = result
    assert output_ids.tolist() == [[10, 11, 20, 99]]
    assert output_tokens == 2
    assert acceptance == [1, 1]
    assert phases["stop_tokens_trimmed"] == 1

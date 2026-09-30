from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Benchmark.common.vllm_pilot import (  # noqa: E402
    domino_greedy_sample,
    install_domino_vllm_compat,
    paired_vllm_metrics,
    pilot_warmup_tokens,
    resolve_eagle3_aux_hidden_state_layers,
    select_pilot_methods,
    speculative_token_count,
)


class PreferLastToken(nn.Module):
    def __init__(self, vocab_size: int, preferred_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = states.new_full((*states.shape[:-1], self.vocab_size), -10.0)
        logits[..., self.preferred_token] = 10.0
        return logits


def test_domino_sampler_leaves_prefix_uncorrected_and_corrects_suffix_per_row():
    batch, block, vocab, hidden = 2, 4, 7, 5
    base_logits = torch.zeros(batch, block, vocab)
    base_logits[0, 0, 1] = 5.0
    base_logits[0, 1, 2] = 5.0
    base_logits[1, 0, 3] = 5.0
    base_logits[1, 1, 4] = 5.0
    target_embeddings = nn.Embedding(vocab, hidden)
    prefix_gru = nn.GRU(hidden, 3, batch_first=True, bias=False)
    correction = PreferLastToken(vocab, preferred_token=6)
    draft_hidden = torch.randn(batch, block, hidden)
    bonus_ids = torch.tensor([5, 0])

    tokens = domino_greedy_sample(
        base_logits,
        draft_hidden,
        bonus_ids,
        target_embeddings,
        prefix_gru,
        correction,
        prefix_len=2,
    )

    assert tokens.shape == (batch, block)
    assert tokens.tolist() == [[1, 2, 6, 6], [3, 4, 6, 6]]


def test_domino_sampler_matches_author_single_sequence_correction_order():
    torch.manual_seed(1729)
    batch, block, vocab, hidden, gru_dim, emb_dim = 2, 16, 23, 7, 5, 6
    base_logits = torch.randn(batch, block, vocab)
    draft_hidden = torch.randn(batch, block, hidden)
    target_embeddings = nn.Embedding(vocab, hidden)
    prefix_gru = nn.GRU(hidden, gru_dim, batch_first=True, bias=False)
    embed_proj = nn.Sequential(
        nn.Linear(hidden + gru_dim, emb_dim, bias=False),
        nn.SiLU(),
        nn.Linear(emb_dim, vocab, bias=False),
    )
    bonus_ids = torch.tensor([3, 17])

    actual = domino_greedy_sample(
        base_logits, draft_hidden, bonus_ids, target_embeddings, prefix_gru,
        embed_proj, prefix_len=1,
    )

    # Direct single-sequence reference following Domino's released spec_generate:
    # take the pure-draft prefix from base logits, prime GRU with [bonus, prefix],
    # then correct each suffix position and advance GRU with that chosen token.
    expected = []
    for row in range(batch):
        row_logits = base_logits[row : row + 1]
        row_hidden = draft_hidden[row : row + 1]
        draft_ids = row_logits.argmax(dim=-1)
        prefix_ids = draft_ids[:, :1]
        realized_prefix = torch.cat((bonus_ids[row : row + 1, None], prefix_ids), dim=1)
        _, gru_state = prefix_gru(target_embeddings(realized_prefix))
        for position in range(1, block):
            state = gru_state.transpose(0, 1)
            z_i = row_hidden[:, position : position + 1]
            correction = embed_proj(torch.cat((z_i, state), dim=-1))
            token = (row_logits[:, position : position + 1] + correction).argmax(dim=-1)
            draft_ids[:, position : position + 1] = token
            if position + 1 < block:
                _, gru_state = prefix_gru(target_embeddings(token), gru_state)
        expected.append(draft_ids[0])

    assert torch.equal(actual, torch.stack(expected))



def test_domino_vllm_adapter_uses_target_lm_head_for_base_logits(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    vocab_size, hidden_size, block_size = 7, 5, 2

    class FakeDFlashModel:
        def __init__(self, config):
            self.config = config
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    class FakeDFlashProposer:
        def __init__(self, model):
            self.model = model
            self.draft_model_config = SimpleNamespace(hf_config=domino_config)
            self.num_speculative_tokens = block_size
            self.use_heterogeneous_vocab = False
            self.load_calls = 0
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size)
            )

        def load_model(self, target_model):
            self.load_calls += 1
            return None

        def set_inputs_first_pass(self, *args, **kwargs):
            return None

        def _sample_draft_tokens(self, hidden_states, sampling_metadata):
            return torch.full((hidden_states.shape[0],), 1), None

    class FakeDFlashSpeculator:
        def load_draft_model(self, target_model, target_attn_layer_names):
            return None

        def _generate_draft(self, *args, **kwargs):
            return None

    class DistinctLogits:
        def __init__(self, token_id):
            self.token_id = token_id
            self.calls = 0

        def __call__(self, hidden_states):
            self.calls += 1
            logits = hidden_states.new_full(
                (hidden_states.shape[0], vocab_size), -10.0
            )
            logits[:, self.token_id] = 10.0
            return logits

    domino_config = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dflash_config={
            "projector_type": "domino",
            "gru_hidden_dim": 3,
            "emb_dim": 4,
            "pure_draft_prefix_len": block_size,
        },
    )
    packages = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.v1",
        "vllm.v1.spec_decode",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.dflash",
    )
    for package in packages:
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    dflash_model_module = ModuleType("vllm.model_executor.models.qwen3_dflash")
    dflash_model_module.DFlashQwen3Model = FakeDFlashModel
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_dflash",
        dflash_model_module,
    )
    dflash_proposer_module = ModuleType("vllm.v1.spec_decode.dflash")
    dflash_proposer_module.DFlashProposer = FakeDFlashProposer
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.spec_decode.dflash",
        dflash_proposer_module,
    )
    speculator_module = ModuleType(
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
    )
    speculator_module.DFlashSpeculator = FakeDFlashSpeculator
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        speculator_module,
    )

    install_domino_vllm_compat()

    draft_core = FakeDFlashModel(domino_config)
    draft_model = SimpleNamespace(
        model=draft_core,
        draft_id_to_target_id=None,
        compute_logits=DistinctLogits(token_id=1),
    )
    proposer = FakeDFlashProposer(draft_model)
    class ForbiddenLMHead:
        def __init__(self):
            self.calls = 0

        def __call__(self, hidden_states):
            self.calls += 1
            raise AssertionError("vLLM LMHead.forward must not be called directly")

    target_lm_head = ForbiddenLMHead()
    target_compute_logits = DistinctLogits(token_id=3)
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=nn.Embedding(vocab_size, hidden_size)),
        lm_head=target_lm_head,
        compute_logits=target_compute_logits,
    )
    proposer.load_model(target_model)
    proposer._domino_bonus_token_ids = torch.tensor([0])

    draft_ids, _ = proposer._sample_draft_tokens(
        torch.zeros(block_size, hidden_size), SimpleNamespace(all_greedy=True)
    )

    assert proposer.load_calls == 1
    assert target_compute_logits.calls == 1
    assert target_lm_head.calls == 0
    assert draft_model.compute_logits.calls == 0
    assert draft_ids.tolist() == [3, 3]



def test_domino_vllm_v2_speculator_applies_domino_correction(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    vocab_size, hidden_size, block_size = 7, 5, 2

    class FakeDFlashModel:
        def __init__(self, config):
            self.config = config
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    class FakeDFlashProposer:
        def __init__(self, model):
            self.model = model
            self.draft_model_config = SimpleNamespace(hf_config=domino_config)
            self.num_speculative_tokens = block_size
            self.use_heterogeneous_vocab = False
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size)
            )

        def load_model(self, target_model):
            return None

        def set_inputs_first_pass(self, *args, **kwargs):
            return None

        def _sample_draft_tokens(self, hidden_states, sampling_metadata):
            return torch.full((hidden_states.shape[0],), 1), None

    class FakeDFlashSpeculator:
        def __init__(self, model):
            self.model = model
            self.draft_model_config = SimpleNamespace(hf_config=domino_config)
            self.num_speculative_steps = block_size
            self.num_query_per_req = block_size + 1
            # vLLM startup profiling can use dummy non-greedy temperatures;
            # the pilot pins actual SamplingParams to temperature=0.
            self.temperature = torch.ones(1)
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size)
            )
            self.sample_indices = torch.tensor([1, 2])
            self.input_buffers = SimpleNamespace(input_ids=torch.tensor([0, 4, 4]))
            self.draft_tokens = torch.zeros((1, block_size), dtype=torch.long)

        def load_draft_model(self, target_model, target_attn_layer_names):
            return self.model

        def _run_model(self, *args, **kwargs):
            return torch.zeros((block_size + 1, hidden_size))

        def _generate_draft(self, *args, **kwargs):
            self.draft_tokens[0].fill_(1)

    class DistinctLogits:
        def __init__(self, token_id):
            self.token_id = token_id
            self.calls = 0

        def __call__(self, hidden_states):
            self.calls += 1
            logits = hidden_states.new_full(
                (hidden_states.shape[0], vocab_size), -10.0
            )
            logits[:, self.token_id] = 10.0
            return logits

    domino_config = SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dflash_config={
            "projector_type": "domino",
            "gru_hidden_dim": 3,
            "emb_dim": 4,
            "pure_draft_prefix_len": block_size,
        },
    )
    packages = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.v1",
        "vllm.v1.spec_decode",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.dflash",
    )
    for package in packages:
        module = ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)

    dflash_model_module = ModuleType("vllm.model_executor.models.qwen3_dflash")
    dflash_model_module.DFlashQwen3Model = FakeDFlashModel
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models.qwen3_dflash",
        dflash_model_module,
    )
    proposer_module = ModuleType("vllm.v1.spec_decode.dflash")
    proposer_module.DFlashProposer = FakeDFlashProposer
    monkeypatch.setitem(sys.modules, "vllm.v1.spec_decode.dflash", proposer_module)
    speculator_module = ModuleType(
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
    )
    speculator_module.DFlashSpeculator = FakeDFlashSpeculator
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
        speculator_module,
    )

    install_domino_vllm_compat()

    draft_core = FakeDFlashModel(domino_config)
    draft_head = DistinctLogits(token_id=1)
    draft_model = SimpleNamespace(
        model=draft_core,
        lm_head=draft_head,
        compute_logits=draft_head,
    )
    speculator = FakeDFlashSpeculator(draft_model)
    class ForbiddenLMHead:
        def __init__(self):
            self.calls = 0

        def __call__(self, hidden_states):
            self.calls += 1
            raise AssertionError("vLLM LMHead.forward must not be called directly")

    target_lm_head = ForbiddenLMHead()
    target_compute_logits = DistinctLogits(token_id=3)
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=nn.Embedding(vocab_size, hidden_size)),
        lm_head=target_lm_head,
        compute_logits=target_compute_logits,
    )

    speculator.load_draft_model(target_model, set())
    speculator._generate_draft(1, block_size + 1, None, None, None)

    assert target_compute_logits.calls == 1
    assert target_lm_head.calls == 0
    assert draft_head.calls == 0
    assert speculator.draft_tokens.tolist() == [[3, 3]]

def test_paired_vllm_metrics_uses_shared_prompts_and_user_dsr_esr_formula():
    records = [
        {"method": "vanilla_vllm", "sample_id": "a", "output_tokens": 10,
         "prefill_ms": 20.0, "mean_itl_ms": 2.0},
        {"method": "vanilla_vllm", "sample_id": "b", "output_tokens": 6,
         "prefill_ms": 30.0, "mean_itl_ms": 4.0},
        {"method": "eagle3", "sample_id": "a", "output_tokens": 8,
         "prefill_ms": 99.0, "mean_itl_ms": 1.0},
        {"method": "eagle3", "sample_id": "b", "output_tokens": 4,
         "prefill_ms": 99.0, "mean_itl_ms": 3.0},
    ]

    rows = paired_vllm_metrics(records, reference="vanilla_vllm")

    result = rows["eagle3"]
    assert result["paired_samples"] == 2
    assert result["reference_prefill_ms"] == pytest.approx(25.0)
    assert result["mean_min_output_tokens"] == pytest.approx(6.0)
    assert result["reference_mean_itl_ms"] == pytest.approx(3.0)
    assert result["method_mean_itl_ms"] == pytest.approx(2.0)
    assert result["dsr"] == pytest.approx(1.5)
    assert result["esr"] == pytest.approx(43.0 / 37.0)


def test_paired_vllm_metrics_rejects_unpaired_sample_sets():
    records = [
        {"method": "vanilla_vllm", "sample_id": "a", "output_tokens": 10,
         "prefill_ms": 20.0, "mean_itl_ms": 2.0},
        {"method": "eagle3", "sample_id": "b", "output_tokens": 8,
         "prefill_ms": 25.0, "mean_itl_ms": 1.0},
    ]

    with pytest.raises(ValueError, match="sample coverage mismatch"):
        paired_vllm_metrics(records, reference="vanilla_vllm")


def test_eagle3_pilot_uses_requested_block_size_16_even_if_config_omits_it():
    assert speculative_token_count(
        "eagle3", {"eagle_config": {"num_lookahead_tokens": 2}}
    ) == 16


def test_eagle3_aux_layer_resolution_matches_vllm_and_deepspec_metadata():
    deep_spec_config = {"target_layer_ids": [1, 9, 17, 25, 33]}
    assert resolve_eagle3_aux_hidden_state_layers(
        deep_spec_config, target_num_hidden_layers=36
    ) == (2, 10, 18, 26, 34)

    angelslim_config = {"architectures": ["Eagle3LlamaForCausalLM"]}
    assert resolve_eagle3_aux_hidden_state_layers(
        angelslim_config, target_num_hidden_layers=36
    ) == (2, 18, 33)

    explicit_config = {"eagle_aux_hidden_state_layer_ids": [3, 11, 27]}
    assert resolve_eagle3_aux_hidden_state_layers(
        explicit_config, target_num_hidden_layers=36
    ) == (3, 11, 27)


def test_pilot_warmup_covers_at_least_one_full_speculative_block():
    assert pilot_warmup_tokens(None) == 64
    assert pilot_warmup_tokens(16) == 64
    assert pilot_warmup_tokens(64) == 65


def test_select_pilot_methods_requires_reference_and_rejects_unknown_methods():
    available = ("vanilla_vllm", "eagle3", "dflash", "domino", "dspark")
    selected = select_pilot_methods(
        "vanilla_vllm,eagle3,dflash,domino", available_methods=available
    )
    assert selected == ("vanilla_vllm", "eagle3", "dflash", "domino")

    with pytest.raises(ValueError, match="must include reference"):
        select_pilot_methods("eagle3,domino", available_methods=available)
    with pytest.raises(ValueError, match="unknown method"):
        select_pilot_methods("vanilla_vllm,eagle3,bogus", available_methods=available)

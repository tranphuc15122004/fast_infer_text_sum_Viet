"""Helpers for the paired common-vLLM pilot, including a Domino DFlash adapter."""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
import math
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch
    from torch import nn


EAGLE3_SPECULATIVE_TOKENS = 16


def select_pilot_methods(
    method_names: str | tuple[str, ...] | list[str],
    *,
    available_methods: tuple[str, ...],
    reference: str = "vanilla_vllm",
) -> tuple[str, ...]:
    """Validate a comma-separated pilot method list and require its reference."""

    if isinstance(method_names, str):
        methods = tuple(part.strip() for part in method_names.split(",") if part.strip())
    else:
        methods = tuple(str(method).strip() for method in method_names if str(method).strip())
    if not methods:
        raise ValueError("at least one pilot method is required")
    if len(set(methods)) != len(methods):
        raise ValueError("pilot method list contains duplicates")
    unknown = sorted(set(methods) - set(available_methods))
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    if reference not in methods:
        raise ValueError(f"pilot method list must include reference {reference!r}")
    return methods


def _positive_config_value(config: Any, *keys: str) -> int | None:
    for key in keys:
        value = config.get(key) if isinstance(config, dict) else getattr(config, key, None)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def speculative_token_count(method: str, config: Any) -> int:
    """Resolve the pilot block size, pinning the requested EAGLE3 setting to 16."""

    if method == "eagle3":
        return EAGLE3_SPECULATIVE_TOKENS

    if isinstance(config, dict):
        raw = config
    elif hasattr(config, "to_dict"):
        raw = config.to_dict()
    else:
        raw = {}
    nested = raw.get("dflash_config") or getattr(config, "dflash_config", {}) or {}
    if method == "dspark":
        value = _positive_config_value(config, "dspark_block_size", "block_size", "n_predict")
        value = value or _positive_config_value(nested, "dspark_block_size", "block_size")
    else:
        value = _positive_config_value(config, "block_size", "n_predict")
        value = value or _positive_config_value(nested, "block_size", "n_predict")
    if value is None:
        raise ValueError(f"could not read speculative block size from {method} config")
    return value



def _pilot_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def resolve_eagle3_aux_hidden_state_layers(
    draft_config: Any, *, target_num_hidden_layers: int
) -> tuple[int, ...] | None:
    """Resolve Eagle3's target hidden-state indices as vLLM 0.30 does.

    The checkpoint's ``target_layer_ids`` refer to decoder layers, while vLLM's
    target feature taps count layer outputs from one, so the former gain +1.
    If a checkpoint provides no IDs, vLLM falls back to the target model's
    standard Eagle3 triplet.
    """

    layer_ids = _pilot_config_value(
        draft_config, "eagle_aux_hidden_state_layer_ids"
    )
    if not layer_ids:
        dflash_config = _pilot_config_value(draft_config, "dflash_config")
        dflash_target_ids = _pilot_config_value(dflash_config, "target_layer_ids")
        if dflash_target_ids:
            layer_ids = [int(layer_id) + 1 for layer_id in dflash_target_ids]
    if not layer_ids:
        dspark_target_ids = _pilot_config_value(
            draft_config, "dspark_target_layer_ids"
        )
        if dspark_target_ids:
            layer_ids = [int(layer_id) + 1 for layer_id in dspark_target_ids]
    if not layer_ids:
        target_layer_ids = _pilot_config_value(draft_config, "target_layer_ids")
        if target_layer_ids:
            layer_ids = [int(layer_id) + 1 for layer_id in target_layer_ids]
    if not layer_ids:
        for config_name in ("dflash_config", "eagle_config"):
            nested_config = _pilot_config_value(draft_config, config_name)
            nested_layer_ids = _pilot_config_value(nested_config, "layer_ids")
            if nested_layer_ids:
                layer_ids = nested_layer_ids
                break
    if layer_ids and isinstance(layer_ids, (list, tuple)):
        return tuple(int(layer_id) for layer_id in layer_ids)

    num_layers = int(target_num_hidden_layers)
    if num_layers < 3:
        return None
    return (2, num_layers // 2, num_layers - 3)


def pilot_warmup_tokens(num_speculative_tokens: int | None) -> int:
    """Return enough unmeasured tokens to exercise speculative blocks before timing."""

    block_size = max(0, int(num_speculative_tokens or 0))
    return max(64, block_size + 1)


def domino_greedy_sample(
    base_logits: "torch.Tensor",
    draft_hidden_states: "torch.Tensor",
    bonus_token_ids: "torch.Tensor",
    embed_tokens: "Callable[[torch.Tensor], torch.Tensor]",
    prefix_gru: "nn.Module",
    embed_proj: "nn.Module",
    *,
    prefix_len: int,
) -> torch.Tensor:
    """Apply Domino's causal logit correction to one parallel draft block.

    ``base_logits`` and ``draft_hidden_states`` use request-major ``[B, K, *]``
    layout. The initial ``prefix_len`` predictions come directly from DFlash;
    the GRU is primed with the target bonus token and those realized drafts,
    then its state corrects each remaining position in order.
    """

    import torch

    if base_logits.ndim != 3 or draft_hidden_states.ndim != 3:
        raise ValueError("Domino expects [batch, block, feature] tensors")
    batch_size, block_size, vocab_size = base_logits.shape
    if draft_hidden_states.shape[:2] != (batch_size, block_size):
        raise ValueError("draft hidden states and logits must share [batch, block]")
    if bonus_token_ids.ndim != 1 or bonus_token_ids.shape[0] != batch_size:
        raise ValueError("bonus token IDs must have shape [batch]")
    if not 0 <= prefix_len <= block_size:
        raise ValueError("prefix_len must be between zero and the draft block size")
    if vocab_size <= 0:
        raise ValueError("draft vocabulary must be non-empty")

    draft_ids = base_logits.argmax(dim=-1)
    prefix_ids = draft_ids[:, :prefix_len]
    realized_prefix = torch.cat((bonus_token_ids[:, None], prefix_ids), dim=1)
    _, gru_state = prefix_gru(embed_tokens(realized_prefix))

    for position in range(prefix_len, block_size):
        state = gru_state.transpose(0, 1)
        correction_input = torch.cat(
            (draft_hidden_states[:, position : position + 1], state), dim=-1
        )
        corrected_logits = base_logits[:, position : position + 1] + embed_proj(
            correction_input
        )
        token_ids = corrected_logits.argmax(dim=-1).squeeze(1)
        draft_ids[:, position] = token_ids

        if position + 1 < block_size:
            _, gru_state = prefix_gru(embed_tokens(token_ids[:, None]), gru_state)

    return draft_ids


def _domino_config(hf_config: Any) -> dict[str, Any]:
    config = getattr(hf_config, "dflash_config", None)
    if config is None and hasattr(hf_config, "to_dict"):
        config = hf_config.to_dict().get("dflash_config")
    if not isinstance(config, dict):
        try:
            config = dict(config or {})
        except (TypeError, ValueError):
            config = {}
    return config


def install_domino_vllm_compat() -> None:
    """Install the checkpoint-specific Domino extension on vLLM's DFlash path.

    The target model, scheduler, DFlash backbone, and rejection verifier remain
    vLLM 0.30 components. This adds the four Domino head tensors to the DFlash
    model, uses the target LM head for base logits like the reference, and
    applies Domino's sequential correction in the parallel sampler.
    Greedy sampling and a full, shared vocabulary are required by this pilot.
    """

    import torch
    from torch import nn
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.v1.spec_decode.dflash import DFlashProposer

    if getattr(DFlashQwen3Model, "_fast_infer_domino_compat", False):
        return

    original_init = DFlashQwen3Model.__init__

    @wraps(original_init)
    def domino_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        config = _domino_config(self.config)
        if config.get("projector_type") != "domino":
            return
        hidden_size = int(self.config.hidden_size)
        gru_hidden_dim = int(config["gru_hidden_dim"])
        embedding_dim = int(config["emb_dim"])
        vocab_size = int(self.config.vocab_size)
        reference_weight = self.embed_tokens.weight
        options = {
            "device": reference_weight.device,
            "dtype": reference_weight.dtype,
        }
        self.prefix_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
            **options,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, embedding_dim, bias=False, **options),
            nn.SiLU(),
            nn.Linear(embedding_dim, vocab_size, bias=False, **options),
        )

    DFlashQwen3Model.__init__ = domino_model_init

    original_load_model = DFlashProposer.load_model

    @wraps(original_load_model)
    def domino_load_model(self: Any, target_model: nn.Module) -> None:
        original_load_model(self, target_model)
        config = _domino_config(self.draft_model_config.hf_config)
        if config.get("projector_type") != "domino":
            return
        while hasattr(target_model, "unwrap"):
            target_model = target_model.unwrap()
        target_core = getattr(target_model, "model", None)
        target_embeddings = getattr(target_core, "embed_tokens", None)
        target_lm_head = getattr(target_model, "lm_head", None)
        target_compute_logits = getattr(target_model, "compute_logits", None)
        if target_embeddings is None:
            raise RuntimeError("could not locate the target model token embedding")
        if target_lm_head is None:
            raise RuntimeError("could not locate the target model LM head")
        if not callable(target_compute_logits):
            raise RuntimeError("could not locate the target model compute_logits method")
        self._domino_target_embed_tokens = target_embeddings
        self._domino_target_lm_head = target_lm_head
        self._domino_target_compute_logits = target_compute_logits

    DFlashProposer.load_model = domino_load_model

    original_set_inputs = DFlashProposer.set_inputs_first_pass

    @wraps(original_set_inputs)
    def domino_set_inputs(self: Any, *args: Any, **kwargs: Any):
        bonus_token_ids = kwargs.get("next_token_ids")
        if bonus_token_ids is None and len(args) >= 2:
            bonus_token_ids = args[1]
        self._domino_bonus_token_ids = bonus_token_ids
        return original_set_inputs(self, *args, **kwargs)

    DFlashProposer.set_inputs_first_pass = domino_set_inputs

    original_sample_draft_tokens = DFlashProposer._sample_draft_tokens

    @wraps(original_sample_draft_tokens)
    def domino_sample_draft_tokens(
        self: Any, hidden_states: "torch.Tensor", sampling_metadata: Any
    ):
        config = _domino_config(self.draft_model_config.hf_config)
        if config.get("projector_type") != "domino":
            return original_sample_draft_tokens(self, hidden_states, sampling_metadata)
        if not sampling_metadata.all_greedy:
            raise NotImplementedError("the Domino pilot requires greedy decoding")
        if self.use_heterogeneous_vocab:
            raise NotImplementedError("the Domino pilot requires a shared full vocabulary")

        bonus_token_ids = getattr(self, "_domino_bonus_token_ids", None)
        if bonus_token_ids is None:
            raise RuntimeError("Domino bonus token IDs were not captured for this block")

        model = self.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        domino_model = model.model
        if getattr(model, "draft_id_to_target_id", None) is not None:
            raise NotImplementedError("Domino checkpoint must use the target vocabulary")

        block_size = int(self.num_speculative_tokens)
        batch_size = int(bonus_token_ids.shape[0])
        if hidden_states.shape[0] != batch_size * block_size:
            raise ValueError(
                "unexpected parallel draft layout: "
                f"{hidden_states.shape[0]} rows for {batch_size} requests x {block_size} tokens"
            )
        target_compute_logits = getattr(self, "_domino_target_compute_logits", None)
        if target_compute_logits is None:
            raise RuntimeError("Domino target compute_logits was not captured during model load")
        logits = target_compute_logits(hidden_states)
        if logits is None or logits.ndim != 2:
            raise ValueError("Domino target LM head must return a [tokens, vocab] tensor")
        if logits.shape[-1] != int(self.vllm_config.model_config.get_vocab_size()):
            raise ValueError("Domino correction head and target vocab sizes differ")
        draft_ids = domino_greedy_sample(
            logits.reshape(batch_size, block_size, -1),
            hidden_states.reshape(batch_size, block_size, -1),
            bonus_token_ids,
            self._domino_target_embed_tokens,
            domino_model.prefix_gru,
            domino_model.embed_proj,
            prefix_len=int(config.get("pure_draft_prefix_len", 0)),
        )
        return draft_ids.reshape(-1), None

    DFlashProposer._sample_draft_tokens = domino_sample_draft_tokens

    # vLLM 0.30 defaults to Model Runner V2 on supported CUDA platforms. Its
    # DFlashSpeculator bypasses DFlashProposer._sample_draft_tokens, so install
    # the same Domino correction at the active V2 generation hook as well.
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if not getattr(DFlashSpeculator, "_fast_infer_domino_compat", False):
        original_v2_load_draft_model = DFlashSpeculator.load_draft_model

        @wraps(original_v2_load_draft_model)
        def domino_v2_load_draft_model(
            self: Any, target_model: nn.Module, *args: Any, **kwargs: Any
        ):
            draft_model = original_v2_load_draft_model(
                self, target_model, *args, **kwargs
            )
            config = _domino_config(self.draft_model_config.hf_config)
            if config.get("projector_type") != "domino":
                return draft_model

            while hasattr(target_model, "unwrap"):
                target_model = target_model.unwrap()
            target_language_model = (
                target_model.get_language_model()
                if hasattr(target_model, "get_language_model")
                else target_model
            )
            target_core = getattr(target_language_model, "model", target_language_model)
            target_embeddings = getattr(target_core, "embed_tokens", None)
            target_lm_head = getattr(target_language_model, "lm_head", None)
            target_compute_logits = getattr(target_model, "compute_logits", None)
            if not callable(target_compute_logits):
                target_compute_logits = getattr(target_language_model, "compute_logits", None)
            if target_embeddings is None:
                raise RuntimeError("could not locate the target model token embedding")
            if target_lm_head is None:
                raise RuntimeError("could not locate the target model LM head")
            if not callable(target_compute_logits):
                raise RuntimeError("could not locate the target model compute_logits method")
            self._domino_target_embed_tokens = target_embeddings
            self._domino_target_lm_head = target_lm_head
            self._domino_target_compute_logits = target_compute_logits
            return draft_model

        DFlashSpeculator.load_draft_model = domino_v2_load_draft_model

        original_v2_generate_draft = DFlashSpeculator._generate_draft

        @wraps(original_v2_generate_draft)
        def domino_v2_generate_draft(
            self: Any,
            num_reqs: int,
            num_tokens_padded: int,
            attn_metadata: Any,
            slot_mappings: Any,
            num_tokens_across_dp: Any,
            cudagraph_runtime_mode: Any = None,
        ) -> None:
            config = _domino_config(self.draft_model_config.hf_config)
            if config.get("projector_type") != "domino":
                return original_v2_generate_draft(
                    self,
                    num_reqs,
                    num_tokens_padded,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cudagraph_runtime_mode,
                )

            import torch

            # This pilot fixes temperature=0 in every measured and warmup
            # SamplingParams. vLLM startup profiling can populate `temperature`
            # with a non-greedy dummy value, so checking it here rejects valid
            # initialization before any request has supplied sampling metadata.

            last_hidden_states = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            block_size = int(self.num_speculative_steps)
            num_sample = int(num_reqs) * block_size
            sample_hidden_states = last_hidden_states[
                self.sample_indices[:num_sample]
            ]
            if sample_hidden_states.shape[0] != num_sample:
                raise ValueError(
                    "unexpected Domino sample layout: "
                    f"{sample_hidden_states.shape[0]} rows for "
                    f"{num_reqs} requests x {block_size} draft tokens"
                )
            query_token_count = int(num_reqs) * int(self.num_query_per_req)
            if self.input_buffers.input_ids.shape[0] < query_token_count:
                raise ValueError("Domino input buffer does not cover every request")
            query_ids = self.input_buffers.input_ids[:query_token_count].reshape(
                int(num_reqs), int(self.num_query_per_req)
            )
            bonus_token_ids = query_ids[:, 0]

            target_compute_logits = getattr(self, "_domino_target_compute_logits", None)
            target_embeddings = getattr(self, "_domino_target_embed_tokens", None)
            if target_compute_logits is None or target_embeddings is None:
                raise RuntimeError(
                    "Domino target embedding and compute_logits were not captured during load"
                )
            logits = target_compute_logits(sample_hidden_states)
            if logits is None or logits.ndim != 2:
                raise ValueError("Domino target LM head must return [tokens, vocab]")
            target_vocab_size = int(
                self.vllm_config.model_config.get_vocab_size()
            )
            if logits.shape[-1] != target_vocab_size:
                raise ValueError("Domino correction head and target vocab sizes differ")

            draft_model = self.model
            while hasattr(draft_model, "unwrap"):
                draft_model = draft_model.unwrap()
            domino_model = draft_model.model
            if getattr(draft_model, "draft_id_to_target_id", None) is not None:
                raise NotImplementedError(
                    "Domino checkpoint must use the target vocabulary"
                )
            draft_ids = domino_greedy_sample(
                logits.reshape(int(num_reqs), block_size, -1),
                sample_hidden_states.reshape(int(num_reqs), block_size, -1),
                bonus_token_ids,
                target_embeddings,
                domino_model.prefix_gru,
                domino_model.embed_proj,
                prefix_len=int(config.get("pure_draft_prefix_len", 0)),
            )
            self.draft_tokens[:num_reqs] = draft_ids
            return None

        DFlashSpeculator._generate_draft = domino_v2_generate_draft
        DFlashSpeculator._fast_infer_domino_compat = True

    DFlashQwen3Model._fast_infer_domino_compat = True


def paired_vllm_metrics(
    records: list[dict[str, Any]], *, reference: str = "vanilla_vllm"
) -> dict[str, dict[str, float | int | str]]:
    """Compute paired DSR/ESR from per-request vLLM output records.

    TPOT is mean per-request ``mean_itl_ms``. The reference's mean prefill
    time is shared across methods; output length uses the per-prompt minimum.
    This is the user-defined formula, separate from the legacy v2 report.
    """

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        method = str(record.get("method", ""))
        sample_id = str(record.get("sample_id", ""))
        if not method or not sample_id:
            raise ValueError("each metric record requires method and sample_id")
        if sample_id in grouped[method]:
            raise ValueError(f"duplicate metric record for {method}/{sample_id}")
        if record.get("status", "success") != "success":
            raise ValueError(f"failed output cannot enter metrics: {method}/{sample_id}")
        grouped[method][sample_id] = record

    if reference not in grouped or not grouped[reference]:
        raise ValueError(f"reference method {reference!r} has no records")
    reference_records = grouped[reference]
    reference_ids = set(reference_records)
    reference_prefills: list[float] = []
    reference_itls: list[float] = []
    for sample_id, record in reference_records.items():
        prefill = _positive_float(record.get("prefill_ms"), "prefill_ms", reference, sample_id)
        itl = _positive_float(record.get("mean_itl_ms"), "mean_itl_ms", reference, sample_id)
        _positive_int(record.get("output_tokens"), "output_tokens", reference, sample_id)
        reference_prefills.append(prefill)
        reference_itls.append(itl)

    common_prefill = sum(reference_prefills) / len(reference_prefills)
    reference_mean_itl = sum(reference_itls) / len(reference_itls)
    output: dict[str, dict[str, float | int | str]] = {}
    for method, method_records in grouped.items():
        if set(method_records) != reference_ids:
            raise ValueError(
                f"sample coverage mismatch for {method}: "
                f"expected {sorted(reference_ids)}, got {sorted(method_records)}"
            )
        method_itls: list[float] = []
        min_lengths: list[int] = []
        for sample_id in sorted(reference_ids):
            record = method_records[sample_id]
            method_itls.append(
                _positive_float(record.get("mean_itl_ms"), "mean_itl_ms", method, sample_id)
            )
            method_tokens = _positive_int(
                record.get("output_tokens"), "output_tokens", method, sample_id
            )
            reference_tokens = int(reference_records[sample_id]["output_tokens"])
            min_lengths.append(min(method_tokens, reference_tokens))

        method_mean_itl = sum(method_itls) / len(method_itls)
        mean_min_tokens = sum(min_lengths) / len(min_lengths)
        dsr = reference_mean_itl / method_mean_itl
        esr = (common_prefill + reference_mean_itl * mean_min_tokens) / (
            common_prefill + method_mean_itl * mean_min_tokens
        )
        output[method] = {
            "paired_samples": len(reference_ids),
            "reference_prefill_ms": common_prefill,
            "reference_mean_itl_ms": reference_mean_itl,
            "method_mean_itl_ms": method_mean_itl,
            "mean_min_output_tokens": mean_min_tokens,
            "dsr": dsr,
            "esr": esr,
        }
    return output


def _positive_float(value: Any, field: str, method: str, sample_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing {field} for {method}/{sample_id}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"invalid {field} for {method}/{sample_id}: {value!r}")
    return parsed


def _positive_int(value: Any, field: str, method: str, sample_id: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing {field} for {method}/{sample_id}") from exc
    if parsed <= 0:
        raise ValueError(f"invalid {field} for {method}/{sample_id}: {value!r}")
    return parsed

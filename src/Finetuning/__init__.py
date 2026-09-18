"""Self-contained SpecForge-compatible DFlash training components."""

from .adaptive_batch import AdaptiveBatchResult, AdaptiveBatchSettings
from .adaptive_inference import AdaptiveInferenceSettings
from .config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .dflash_family_model import OnlineDFlashModel
from .distributed import DistributedContext
from .evaluation import Evaluator
from .model import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    apply_rotary_pos_emb,
    build_target_layer_ids,
    extract_context_feature,
    normalize_draft_head_checkpoint_keys,
    resolve_dflash_attention_layout,
    sample,
)
from .strategy import DFlashTrainStrategy, StepContext, StepOutput, TrainBatch
from .trainer import Trainer

__all__ = [
    "DEFAULT_DFLASH_KERNELS",
    "AdaptiveBatchResult",
    "AdaptiveBatchSettings",
    "AdaptiveInferenceSettings",
    "DataConfig",
    "DFlashKernels",
    "DFlashDraftModel",
    "DFlashTrainStrategy",
    "DistributedContext",
    "Evaluator",
    "ModelConfig",
    "OnlineDFlashModel",
    "RunConfig",
    "StepContext",
    "StepOutput",
    "Trainer",
    "TrainBatch",
    "TrainingConfig",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "apply_rotary_pos_emb",
    "build_target_layer_ids",
    "extract_context_feature",
    "normalize_draft_head_checkpoint_keys",
    "resolve_dflash_attention_layout",
    "sample",
]

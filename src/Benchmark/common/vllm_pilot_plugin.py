"""vLLM general plugin entry point for Domino compatibility."""


def register() -> None:
    """Patch vLLM's DFlash path for the Domino checkpoint in every worker."""

    from .vllm_pilot import install_domino_vllm_compat

    install_domino_vllm_compat()

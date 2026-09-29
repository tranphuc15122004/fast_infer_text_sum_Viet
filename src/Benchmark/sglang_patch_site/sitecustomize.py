"""Bootstrap SGLang timing only for its benchmark server subprocess."""

import os

if os.environ.get("FAST_INFER_SGLANG_TIMING_PATCH") == "1":
    from Benchmark.sglang_timing_patch import install

    install()

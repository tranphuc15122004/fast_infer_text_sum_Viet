from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_benchmark_runtime_is_importable_from_src_package() -> None:
    from Benchmark.common.longbench_adapter import BASELINES
    from Benchmark.run_longbench_200 import _parser

    assert BASELINES == (
        "vanilla_hf",
        "vanilla_fa",
        "eagle3",
        "dflash",
        "domino",
        "dspark",
    )
    assert _parser().parse_args(["--mode", "smoke"]).mode == "smoke"


def test_shell_launcher_delegates_to_benchmark_package() -> None:
    launcher = (ROOT / "scripts" / "run_longbench_200.sh").read_text(
        encoding="utf-8"
    )
    assert "-m Benchmark.run_longbench_200" in launcher

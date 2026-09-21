from __future__ import annotations

from collections import defaultdict


def test_server_pool_preserves_order_and_retries_on_another_endpoint() -> None:
    from Finetuning.server_pool import OpenAICompatibleServerPool

    attempts: defaultdict[tuple[str, str], int] = defaultdict(int)

    class FakePool(OpenAICompatibleServerPool):
        def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
            request_id = str(payload["request_id"])
            attempts[(endpoint, request_id)] += 1
            if request_id == "retry" and endpoint.endswith("30000/v1"):
                raise RuntimeError("endpoint unavailable")
            return {
                "choices": [{"message": {"content": f"summary-{request_id}"}}]
            }

    pool = FakePool(
        ["http://127.0.0.1:30000/v1", "http://127.0.0.1:30001/v1"],
        model="qwen3",
        concurrency_per_server=2,
        max_retries=1,
        retry_backoff_seconds=0,
    )
    result = pool.generate_many(
        [
            {"request_id": "retry", "messages": []},
            {"request_id": "slow", "messages": []},
            {"request_id": "fast", "messages": []},
        ]
    )

    assert result == ["summary-retry", "summary-slow", "summary-fast"]
    assert attempts[("http://127.0.0.1:30000/v1", "retry")] == 1
    assert attempts[("http://127.0.0.1:30001/v1", "retry")] == 1

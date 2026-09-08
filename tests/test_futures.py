from __future__ import annotations

import asyncio
import time

import pytest
from tinker import types
from tinker.types.try_again_response import TryAgainResponse

from tuft.exceptions import (
    FutureCancelledException,
    ServerException,
    UnknownModelException,
    UserMismatchException,
)
from tuft.futures import FutureStore


async def _wait_for_result(
    store: FutureStore, request_id: str, user_id: str, timeout: float = 10.0
):
    # Generous budget: operations below sleep ~1s, and thread-pool startup or
    # persistence writes can add latency on slow CI runners.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await store.retrieve(request_id, user_id=user_id, timeout=timeout)
        if not isinstance(result, TryAgainResponse):
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("future did not complete in time")


@pytest.mark.asyncio
async def test_future_store_mismatches_user():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    future = await store.enqueue(_operation, model_id="run", user_id="tester")
    with pytest.raises(UserMismatchException) as exc_info:
        await store.retrieve(future.request_id, user_id="wrong_user", timeout=1.0)
    assert "You do not have permission" in str(exc_info.value)
    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_returns_try_again_until_ready():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        time.sleep(1)
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    future = await store.enqueue(_operation, model_id="run", user_id="tester")
    first_response = await store.retrieve(future.request_id, user_id="tester", timeout=0.1)
    assert isinstance(first_response, TryAgainResponse)

    final = await _wait_for_result(store, future.request_id, user_id="tester")
    assert isinstance(final, types.SaveWeightsResponse)
    assert final.path.endswith("ckpt")
    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_records_failures_as_request_failed():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        raise UnknownModelException("unknown")

    future = await store.enqueue(_operation, user_id="tester")
    with pytest.raises(UnknownModelException):
        await _wait_for_result(store, future.request_id, user_id="tester")

    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_handles_unexpected_errors():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:  # pragma: no cover - executed in thread
        raise RuntimeError("boom")

    future = await store.enqueue(_operation, user_id="tester")

    with pytest.raises(ServerException) as exc_info:
        await _wait_for_result(store, future.request_id, user_id="tester")
    assert "boom" in str(exc_info.value)

    await store.shutdown()


@pytest.mark.asyncio
async def test_mark_pending_sample_futures_failed():
    """Test that mark_pending_sample_futures_failed only marks sample futures as failed."""
    store = FutureStore()

    async def _sample_operation() -> types.SampleResponse:
        await asyncio.sleep(10)  # Long enough to stay pending
        return types.SampleResponse(sequences=[])

    async def _training_operation() -> types.SaveWeightsResponse:
        await asyncio.sleep(10)
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    sample_future = await store.enqueue(
        _sample_operation, user_id="tester", operation_type="sample"
    )
    training_future = await store.enqueue(
        _training_operation, user_id="tester", model_id="model-1", operation_type="save_weights"
    )

    sample_result = await store.retrieve(sample_future.request_id, user_id="tester", timeout=0.1)
    training_result = await store.retrieve(
        training_future.request_id, user_id="tester", timeout=0.1
    )
    assert isinstance(sample_result, TryAgainResponse)
    assert isinstance(training_result, TryAgainResponse)

    # Mark sample futures as failed
    count = store.mark_pending_sample_futures_failed()
    assert count == 1  # Only the sample future should be marked

    # Sample future should now be failed
    with pytest.raises(FutureCancelledException):
        sample_result = await store.retrieve(
            sample_future.request_id, user_id="tester", timeout=0.1
        )

    training_result = await store.retrieve(
        training_future.request_id, user_id="tester", timeout=0.1
    )
    # It should NOT be a RequestFailedResponse from our mark call
    assert isinstance(training_result, TryAgainResponse)

    await store.shutdown()


@pytest.mark.asyncio
async def test_completed_future_is_retrievable_after_restart():
    """Completed futures are loaded from Redis on demand, not restored eagerly."""
    store = FutureStore()
    future = await store.create_ready_future(
        types.SaveWeightsResponse(path="tinker://run/weights/ckpt"),
        user_id="tester",
        model_id="run",
    )
    await store.shutdown()

    restarted = FutureStore()
    try:
        result = await restarted.retrieve(future.request_id, user_id="tester", timeout=0.1)
        # Redis round-trips the payload as JSON, so it comes back as a dict;
        # the HTTP layer passes dicts through unchanged, giving the same
        # response body as a live future.
        assert result["path"] == "tinker://run/weights/ckpt"
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_pending_future_from_previous_process_is_failed():
    """A pending future outlived its process, so nothing can ever complete it."""
    from tuft.futures import FutureRecord
    from tuft.persistence import get_redis_store, save_record

    stale = FutureRecord(
        request_id="stale-request",
        future_id=1,
        user_id="tester",
        status="pending",
        operation_type="sample",
    )
    key = get_redis_store().build_key(FutureStore.REDIS_KEY_PREFIX, "stale-request")
    save_record(key, stale)

    store = FutureStore()
    try:
        with pytest.raises(FutureCancelledException):
            await store.retrieve("stale-request", user_id="tester", timeout=0.1)
    finally:
        await store.shutdown()


@pytest.mark.asyncio
async def test_future_id_counter_survives_restart():
    """IDs stay monotonic across restarts without scanning every future."""
    store = FutureStore()
    await store.create_ready_future(
        types.SaveWeightsResponse(path="tinker://run/weights/ckpt"),
        user_id="tester",
        model_id="run",
    )
    first_id = store.get_current_future_id()
    await store.shutdown()

    restarted = FutureStore()
    try:
        await restarted.create_ready_future(
            types.SaveWeightsResponse(path="tinker://run/weights/ckpt2"),
            user_id="tester",
            model_id="run",
        )
        assert restarted.get_current_future_id() > first_id
    finally:
        await restarted.shutdown()

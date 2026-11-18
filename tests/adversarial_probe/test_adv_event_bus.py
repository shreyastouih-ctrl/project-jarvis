"""Adversarial Concurrency & Stress Probe: Event Bus."""

import asyncio
import random
import time
import pytest
from jarvis.core.event_bus import Event, EventBus, EventType


class TestAdversarialEventBus:
    """Stress tests and concurrency edge cases for EventBus."""

    @pytest.mark.asyncio
    async def test_high_throughput_event_flooding(self):
        """Stress: 10 concurrent publishers firing 100 events each (1000 events total)."""
        bus = EventBus(max_history=2000)
        received_counts = {"specific": 0, "wildcard": 0}

        async def specific_handler(event: Event):
            received_counts["specific"] += 1

        async def wildcard_handler(event: Event):
            received_counts["wildcard"] += 1

        bus.subscribe(EventType.TELEMETRY_UPDATE, specific_handler)
        bus.subscribe(None, wildcard_handler)

        async def publisher(pub_id: int, count: int):
            for i in range(count):
                await bus.publish_event(
                    EventType.TELEMETRY_UPDATE,
                    pub_id=pub_id,
                    seq=i,
                    cpu=random.random() * 100,
                )
                if i % 20 == 0:
                    await asyncio.sleep(0.001)

        t0 = time.perf_counter()
        tasks = [asyncio.create_task(publisher(p, 100)) for p in range(10)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        assert received_counts["specific"] == 1000, f"Expected 1000 specific, got {received_counts['specific']}"
        assert received_counts["wildcard"] == 1000, f"Expected 1000 wildcard, got {received_counts['wildcard']}"
        assert len(bus.get_history()) == 1000
        assert elapsed < 3.0, f"Flooding 1000 events took too long: {elapsed:.2f}s"
        await bus.close()

    @pytest.mark.asyncio
    async def test_dynamic_subscription_and_unsubscription_during_dispatch(self):
        """Edge Case: Subscribers subscribing and unsubscribing during active event dispatch."""
        bus = EventBus()
        call_log = []

        def ephemeral_handler(event: Event):
            call_log.append("ephemeral")
            # Unsubscribe self inside handler
            bus.unsubscribe(EventType.STATE_CHANGED, ephemeral_handler)

        def spawning_handler(event: Event):
            call_log.append("spawner")
            # Subscribe a new handler inside handler
            def spawned(ev: Event):
                call_log.append("spawned")
            bus.subscribe(EventType.STATE_CHANGED, spawned)

        bus.subscribe(EventType.STATE_CHANGED, ephemeral_handler)
        bus.subscribe(EventType.STATE_CHANGED, spawning_handler)

        # First publish: both ephemeral and spawner should run
        await bus.publish_event(EventType.STATE_CHANGED, step=1)
        assert "ephemeral" in call_log
        assert "spawner" in call_log

        # Second publish: ephemeral should NOT run (unsubscribed), spawned SHOULD run
        call_log.clear()
        await bus.publish_event(EventType.STATE_CHANGED, step=2)
        assert "ephemeral" not in call_log
        assert "spawner" in call_log
        assert "spawned" in call_log
        await bus.close()

    @pytest.mark.asyncio
    async def test_unhandled_subscriber_exceptions_isolation(self):
        """Stress: Multiple crashing handlers (sync, async, diverse exceptions)."""
        bus = EventBus()
        survivors = []

        def failing_sync_handler(event: Event):
            raise RuntimeError("Fatal sync error in handler")

        async def failing_async_handler(event: Event):
            await asyncio.sleep(0.001)
            raise ValueError("Fatal async error in handler")

        def zero_div_handler(event: Event):
            _ = 1 / 0

        async def healthy_async_handler(event: Event):
            survivors.append("healthy_async")

        def healthy_sync_handler(event: Event):
            survivors.append("healthy_sync")

        bus.subscribe(EventType.CALL_INCOMING, failing_sync_handler)
        bus.subscribe(EventType.CALL_INCOMING, failing_async_handler)
        bus.subscribe(EventType.CALL_INCOMING, zero_div_handler)
        bus.subscribe(EventType.CALL_INCOMING, healthy_async_handler)
        bus.subscribe(EventType.CALL_INCOMING, healthy_sync_handler)

        # Publishing must not raise or swallow surviving handlers
        await bus.publish_event(EventType.CALL_INCOMING, call_id="c1", caller="Tony")

        assert "healthy_async" in survivors
        assert "healthy_sync" in survivors
        assert len(survivors) == 2
        await bus.close()

    @pytest.mark.asyncio
    async def test_wait_for_concurrency_and_timeout_cleanup(self):
        """Stress: Multiple concurrent waiters resolving and timing out."""
        bus = EventBus()

        async def waiter_success(idx: int):
            event = await bus.wait_for(
                EventType.TRANSCRIPT_READY,
                timeout=1.0,
                predicate=lambda e: e.get("idx") == idx,
            )
            return event.get("idx")

        async def waiter_timeout():
            try:
                await bus.wait_for(EventType.TRANSCRIPT_READY, timeout=0.05)
                return "unexpected_success"
            except TimeoutError:
                return "timeout_ok"

        # Start 5 success waiters and 5 timeout waiters
        success_tasks = [asyncio.create_task(waiter_success(i)) for i in range(5)]
        timeout_tasks = [asyncio.create_task(waiter_timeout()) for _ in range(5)]

        # Wait for timeout tasks to fail
        timeout_results = await asyncio.gather(*timeout_tasks)
        assert all(r == "timeout_ok" for r in timeout_results)

        # Now fulfill the success tasks
        for i in range(5):
            await bus.publish_event(EventType.TRANSCRIPT_READY, idx=i)

        success_results = await asyncio.gather(*success_tasks)
        assert success_results == [0, 1, 2, 3, 4]

        # Waiters queue should now be empty
        assert len(bus._waiters) == 0
        await bus.close()

    @pytest.mark.asyncio
    async def test_wait_for_predicate_exception_behavior(self):
        """Vulnerability probe: Predicate inside wait_for raises an unhandled exception during publish."""
        bus = EventBus()
        subscriber_executed = False

        def normal_subscriber(ev: Event):
            nonlocal subscriber_executed
            subscriber_executed = True

        bus.subscribe(EventType.INTENT_DISPATCHED, normal_subscriber)

        def buggy_predicate(e: Event):
            raise KeyError("Corrupt event payload check in predicate")

        waiter_task = asyncio.create_task(
            bus.wait_for(EventType.INTENT_DISPATCHED, timeout=0.1, predicate=buggy_predicate)
        )
        await asyncio.sleep(0.01)

        # Publish matching event: we observe whether publish_event catches or propagates predicate error
        publish_exception = None
        try:
            await bus.publish_event(EventType.INTENT_DISPATCHED, intent="test")
        except Exception as exc:
            publish_exception = exc

        # Collect waiter task result
        try:
            await waiter_task
        except Exception:
            pass

        await bus.close()

        # Verification: publish_event must NOT crash due to waiter predicate exception,
        # and normal subscribers must execute cleanly
        assert publish_exception is None, f"publish_event crashed with exception: {publish_exception}"
        assert subscriber_executed is True, "Normal subscriber was not executed due to predicate exception"

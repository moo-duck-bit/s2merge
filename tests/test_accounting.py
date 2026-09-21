"""Token usage is attributed to the stage that spent it."""

import asyncio

import pytest

from s2rmerge import accounting
from s2rmerge.accounting import Ledger, Usage


@pytest.fixture()
def ledger():
    return Ledger()


def test_usage_starts_empty(ledger):
    assert ledger.total == Usage()


def test_calls_land_in_the_active_stage(ledger):
    with ledger.stage(accounting.ROUTER):
        ledger.record(1000, 500, cost=0.01)
    with ledger.stage(accounting.INFERENCE):
        ledger.record(10000, 4000, cost=0.2)

    assert ledger[accounting.ROUTER].prompt_tokens == 1000
    assert ledger[accounting.INFERENCE].prompt_tokens == 10000
    assert ledger[accounting.OPTIMIZATION].total_tokens == 0

    assert ledger.total.prompt_tokens == 11000
    assert ledger.total.completion_tokens == 4500
    assert ledger.total.cost == pytest.approx(0.21)


def test_stages_nest_and_restore(ledger):
    with ledger.stage(accounting.OPTIMIZATION):
        with ledger.stage(accounting.FUSION):
            ledger.record(331, 91)
        ledger.record(100, 10)

    assert ledger[accounting.FUSION].prompt_tokens == 331
    assert ledger[accounting.OPTIMIZATION].prompt_tokens == 100


def test_attribution_survives_concurrent_tasks(ledger):
    """A graph runs its queries under asyncio.gather, so the stage has to
    propagate into every spawned task rather than leak between them."""

    async def record_one(prompt_tokens):
        await asyncio.sleep(0)
        ledger.record(prompt_tokens, 0)

    async def main():
        with ledger.stage(accounting.INFERENCE):
            await asyncio.gather(*(record_one(n) for n in (1, 2, 3, 4)))

    asyncio.run(main())
    assert ledger[accounting.INFERENCE].prompt_tokens == 10


def test_unknown_stage_is_rejected(ledger):
    with pytest.raises(ValueError):
        with ledger.stage("not-a-stage"):
            pass


def test_reset_clears_every_stage(ledger):
    with ledger.stage(accounting.ROUTER):
        ledger.record(5, 5, cost=1.0)
    ledger.reset()
    assert ledger.total == Usage()

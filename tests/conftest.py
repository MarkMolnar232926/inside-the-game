import pytest

from pitchlens.synth.generate import generate


@pytest.fixture(scope="session")
def short_match():
    """A 20-minute synthetic match (fast; enough for detectors, agents and rates)."""
    return generate(seed=4, narrative="late_comeback", duration_s=1200.0)

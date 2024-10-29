import pytest
import types


from benchmark.baselines.direct_prompt import DirectPrompt
from benchmark.baselines.lag_llama import lag_llama
from benchmark.baselines.llm_processes import LLMPForecaster
from benchmark.baselines.naive import oracle_baseline, random_baseline
from benchmark.baselines.statsmodels import (
    ETSModelForecaster,
    ExponentialSmoothingForecaster,
)
from benchmark.utils import get_all_parent_classes


ALL_BASELINES = [
    DirectPrompt,
    lag_llama,
    LLMPForecaster,
    oracle_baseline,
    random_baseline,
    ETSModelForecaster,
    ExponentialSmoothingForecaster,
]


@pytest.mark.parametrize("baseline", ALL_BASELINES)
def test_version(baseline):
    """
    Test that the baseline defines a version attribute

    """
    assert (
        "__version__" in baseline.__dict__
    ), f"{baseline} should define a __version__ attribute"

    # If the baseline is a class, check that all parent classes define a version attribute
    if isinstance(baseline, type):
        parents = get_all_parent_classes(baseline)
        status = {t: "__version__" in t.__dict__ for t in parents}
        assert all(
            status.values()
        ), f"All parents of {baseline} should define a __version__ attribute but found {status}"

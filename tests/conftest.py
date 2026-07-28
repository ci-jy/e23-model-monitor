import numpy as np
import pytest

from e23mon.data import TARGET, score, synthetic_credit, train_models


@pytest.fixture(scope="session")
def scored_population():
    pop = synthetic_credit(20_000, seed=5)
    features = [c for c in pop.columns if c != TARGET]
    models = train_models(pop.iloc[:5000], features, seed=0)
    return score(models, pop.iloc[5000:]).reset_index(drop=True)


@pytest.fixture
def rng():
    return np.random.default_rng(123)

import pytest

from healthmania.synthetic import make_synthetic


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic(n=240, seed=1)


@pytest.fixture()
def raw_csv(tmp_path, raw_df):
    path = tmp_path / "sleep.csv"
    raw_df.to_csv(path, index=False)
    return path

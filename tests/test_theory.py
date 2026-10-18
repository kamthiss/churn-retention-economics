import pytest

from churn.theory import theory

EXPECTED = {
    60: {"cat_food": 0.213, "dog_food": 0.502, "accessories": 0.623, "total": 0.446},
    90: {"cat_food": 0.019, "dog_food": 0.109, "accessories": 0.453, "total": 0.179},
    120: {"cat_food": 0.001, "dog_food": 0.012, "accessories": 0.283, "total": 0.077},
    180: {"cat_food": 0.000, "dog_food": 0.000, "accessories": 0.078, "total": 0.017},
}
MEDIANS = {"cat_food": 28.1, "dog_food": 37.5, "accessories": 50.1}


@pytest.fixture(scope="module")
def result():
    return theory()


@pytest.mark.parametrize("n", EXPECTED)
def test_false_churn(result, n):
    row = result[0].set_index("N").loc[n]
    for col, val in EXPECTED[n].items():
        assert row[col] == pytest.approx(val, abs=0.015), col


def test_medians(result):
    for cat, val in MEDIANS.items():
        assert result[1][cat] == pytest.approx(val, abs=1), cat

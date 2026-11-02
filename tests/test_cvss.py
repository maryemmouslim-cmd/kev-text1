import pytest
from src import cvss

@pytest.mark.parametrize("vec,score", [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
    ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
    ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0),
    ("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
])
def test_base_score(vec, score):
    assert cvss.base_score(vec) == score

def test_roundup_float_artifact():
    assert cvss.roundup(4.00001) == 4.1 and cvss.roundup(4.000001) == 4.0 and cvss.roundup(4.0) == 4.0

def test_bad_vector():
    assert cvss.try_base_score("CVSS:2.0/AV:N") is None
    with pytest.raises(ValueError):
        cvss.base_score("nonsense")

import rombus._core.integrals as integrals


def test_rate_to_num():
    x = integrals.rate_to_num(0, 10, 50)
    assert x is not None

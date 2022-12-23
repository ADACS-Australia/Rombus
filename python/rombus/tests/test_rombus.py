import rombus

def test_rate_to_num():
    x = rombus.integrals.rate_to_num(0,10,50)
    assert x is not None

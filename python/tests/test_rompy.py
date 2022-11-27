from greedy_mpi import rompy

def test_rate_to_num():
    x = rompy.integrals.rate_to_num(0,10,50)
    assert x is not None

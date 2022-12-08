import pytest
from rombus.scripts.cli import cli
from click.testing import CliRunner
import numpy as np

def test_lal(tmp_path):

    atol = 1e-12
    rtol = 1e-12

    runner = CliRunner()    
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli,["make-reduced-basis", "/Users/gpoole/my_code/rombus/python/tests/data/GreedyPoints.npy"])
        assert result.exit_code == 0
        result = runner.invoke(cli,["make-empirical-interpolant"])
        assert result.exit_code == 0
        result = runner.invoke(cli,["compare-rom-to-true"])
        assert result.exit_code == 0
        delta_diff = np.load("ROM_diff.npy")

    assert np.allclose(delta_diff, 0., rtol=rtol, atol=atol)

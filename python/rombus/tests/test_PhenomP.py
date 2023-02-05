import pytest
from rombus.scripts.cli import cli
from click.testing import CliRunner
import numpy as np


@pytest.mark.lalsuite
def test_PhenomP(tmp_path):

    atol = 1e-12
    rtol = 1e-12

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            [
                "rombus.tests.PhenomP:model",
                "make-reduced-basis",
                "/Users/gpoole/my_code/rombus/python/rombus/tests/data/LALSuite_test_grid.npy",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli, ["rombus.tests.PhenomP:model", "make-empirical-interpolant"]
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli, ["rombus.tests.PhenomP:model", "compare-rom-to-true"]
        )
        assert result.exit_code == 0
        delta_diff = np.load("ROM_diff.npy")

    assert np.allclose(delta_diff, 0.0, rtol=rtol, atol=atol)

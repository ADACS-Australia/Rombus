import numpy as np
import pkg_resources
import pytest
from click.testing import CliRunner

from rombus.scripts.cli import cli


@pytest.mark.lalsuite
def test_PhenomP(tmp_path):

    atol = 1e-12
    rtol = 1e-12

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        greedy_filename = pkg_resources.resource_filename(
            "rombus.tests.resources", "LALSuite_test_grid.npy"
        )
        result = runner.invoke(
            cli,
            [
                "rombus.tests.PhenomP:model",
                "build",
                greedy_filename,
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "rombus.tests.PhenomP:model",
                "evaluate",
                "m1=23.07487209351506",
                "m2=29.753779794217984",
                "chi1L=0.3803919479347709",
                "chi2L=0.266269755913278",
                "chip=0.5458029422691579",
                "thetaJ=1.7621026471916568",
                "alpha=2.765791991075064",
            ],
        )
        assert result.exit_code == 0
        delta_diff = np.load("ROM_diff.npy")

    assert np.allclose(delta_diff, 0.0, rtol=rtol, atol=atol)

from pathlib import Path

import h5py
import numpy as np
import pytest

from yadi import AMIData


@pytest.fixture
def synthetic_ami(tmp_path: Path) -> tuple[Path, int, int]:
    """Synthesize a small AMI HDF5 with the keys AMIData expects."""
    n_nodes, n_steps = 4, 24
    rng = np.random.default_rng(0)
    path = tmp_path / "ami.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("AMI_PkW", data=rng.normal(size=(n_nodes, n_steps)))
        f.create_dataset("AMI_QkVAR", data=rng.normal(size=(n_nodes, n_steps)))
        f.create_dataset("AMI_Vpu", data=1.0 + 0.01 * rng.normal(size=(n_nodes, n_steps)))
        f.create_dataset("loadNames", data=np.array([f"load{i}".encode() for i in range(n_nodes)]))
        f.create_dataset("HC_Vconstrained", data=rng.normal(size=(n_nodes, 2)))
    return path, n_nodes, n_steps


def test_ami_get_datasets_no_interpolate(synthetic_ami):
    path, n_nodes, n_steps = synthetic_ami
    ami = AMIData(str(path), nodes=list(range(n_nodes)), time_steps=list(range(n_steps)))
    data = ami.get_datasets(interpolate=None)
    assert {"P", "Q", "V"}.issubset(data.keys())
    assert data["P"].shape == (n_steps, n_nodes)
    assert data["Q"].shape == (n_steps, n_nodes)
    assert data["V"].shape == (n_steps, n_nodes)


def test_ami_get_datasets_default_interpolate_drops_first_row(synthetic_ami):
    """The default interpolate='linear' path runs `interpolate()` which trims one row."""
    path, n_nodes, n_steps = synthetic_ami
    ami = AMIData(str(path), nodes=list(range(n_nodes)), time_steps=list(range(n_steps)))
    data = ami.get_datasets()
    assert data["V"].shape == (n_steps - 1, n_nodes)
    assert data["P"].shape == (n_steps - 1, n_nodes)
    assert data["Q"].shape == (n_steps - 1, n_nodes)


def test_ami_static_differentiate_shape_and_finite():
    rng = np.random.default_rng(1)
    M, N = 16, 3
    V = 1.0 + 0.01 * rng.normal(size=(M, N))
    P = rng.normal(size=(M, N))
    Q = rng.normal(size=(M, N))
    diffs = AMIData.static_differentiate(V, P, Q)
    assert diffs["DV"].shape == (M, N)
    assert np.isfinite(diffs["DV"]).all()
    assert np.isfinite(diffs["DP"]).all()
    assert np.isfinite(diffs["DQ"]).all()

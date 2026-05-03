import os
from pathlib import Path

import pytest

from yadi import AMIData

AMI_FIXTURE = Path(os.environ.get("YADI_AMI_FIXTURE", "tests/fixtures/ami_sample.h5"))


@pytest.mark.skipif(not AMI_FIXTURE.exists(), reason="AMI HDF5 fixture not present")
def test_ami_loads_datasets() -> None:
    ami = AMIData(str(AMI_FIXTURE))
    data = ami.get_datasets(interpolate=None)
    assert {"P", "Q", "V"}.issubset(data.keys())
    assert data["V"].shape == data["P"].shape == data["Q"].shape

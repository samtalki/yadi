from yadi import DSS_VC_HCA


def test_iterative_hc_drives_voltage_to_cap(ieee13: str) -> None:
    hc = DSS_VC_HCA(
        ieee13,
        verbose=False,
        v_max=1.07,
        kw_inj_max=4000,
        delta_kw_inj=200,
    )
    result = hc.get_iterative_hc()
    assert result.shape == (41,)
    # Positive kW of injection headroom; 0 means the bus stayed below v_max for
    # the whole sweep or was already over the cap.
    assert (result >= 0).all()
    # The IEEE 13 case with a 4 MW sweep should put at least one bus over 1.07 pu.
    assert (result > 0).any(), "no node ever crossed v_max — sweep range too small"

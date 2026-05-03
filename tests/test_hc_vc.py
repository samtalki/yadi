from yadi import DSS_VC_HCA


def test_iterative_hc_returns_vector(case3_balanced: str) -> None:
    hc = DSS_VC_HCA(
        case3_balanced,
        verbose=False,
        kw_inj_max=10,
        delta_kw_inj=2,
    )
    result = hc.get_iterative_hc()
    assert result.shape == (9,)
    # The injection sweep uses negative kw values, so the recorded values are non-positive.
    assert (result <= 0).all()

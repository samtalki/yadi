from yadi import DSS_Data


def test_compile_case3(case3_balanced: str) -> None:
    d = DSS_Data(case3_balanced, verbose=False)
    assert d.dss.Circuit.Name() == "3bus_example"
    assert len(d.dss.Circuit.YNodeOrder()) == 9
    assert d.dss.Solution.Converged()


def test_compile_ieee13(ieee13: str) -> None:
    d = DSS_Data(ieee13, verbose=False)
    name = d.dss.Circuit.Name()
    assert name and "ieee13" in name.lower().replace("_", "")
    assert len(d.dss.Circuit.YNodeOrder()) == 41
    assert d.dss.Solution.Converged()

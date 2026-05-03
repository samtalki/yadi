import yadi.dss.line_code as line_code


class DSS_VoltageSource(line_code.DSS_LineCode):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, precompile=precompile, verbose=verbose)

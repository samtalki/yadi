import yadi.dss.load as load


class DSS_LineCode(load.DSS_Load):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, precompile=precompile, verbose=verbose)

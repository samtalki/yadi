import yadi.dss.transformer as transformer


class DSS_Shunt(transformer.DSS_Transformer):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, precompile=precompile, verbose=verbose)

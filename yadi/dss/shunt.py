import yadi.dss.transformer as transformer


class DSS_Shunt(transformer.DSS_Transformer):
    def __init__(self, redirects, precompile, verbose=False):
        """ "
        Class for handling shunts in OpenDSS.

        """

        super().__init__(redirects, redirects, precompile)

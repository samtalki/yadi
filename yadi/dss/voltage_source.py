import yadi.dss.line_code as line_code


class DSS_VoltageSource(line_code.DSS_LineCode):
    def __init__(self, redirects, precompile, verbose=False):
        """ "
        Class for handling voltage sources in OpenDSS.

        """
        super().__init__(redirects, redirects, precompile)

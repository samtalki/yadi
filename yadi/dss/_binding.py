# Single point of truth for the DSS binding. Migrating to py_dss_interface
# means rewriting this file plus the bodies of the DSS_* wrapper methods
# in yadi/dss/*.py. Nothing else in yadi imports the binding directly.
import opendssdirect as dss

__all__ = ["dss"]

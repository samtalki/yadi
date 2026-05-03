"""Load HDF5 AMI (P, Q, V) matrices."""

import warnings

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


class AMIData:
    """Tabular AMI data: three (M timesteps, N nodes) matrices for P, Q, and V."""

    def __init__(
        self,
        data_path: str,
        nodes: list[int] | None = None,
        time_steps: list[int] | None = None,
    ) -> None:
        if nodes is None:
            nodes = list(range(1379))
        if time_steps is None:
            time_steps = list(range(35040))

        self._data_path = data_path
        self._raw_data = None

        self.data: dict[str, np.ndarray] = {}
        self.hc_baseline: dict[str, np.ndarray] = {}
        self.capab_baseline: dict[str, np.ndarray] = {}

        self.nodal_ami_dfs: list[pd.DataFrame] | None = None
        self.daytime_mask = None

        self.nodes = nodes
        self.N_nodes: int = len(nodes)
        self.feature_index: np.ndarray | None = None

        self.time_steps = time_steps
        self.T_steps: int = len(time_steps)
        self.datetime_index = pd.date_range("2020", freq="15min", periods=self.T_steps)

    def get_datasets(
        self, interpolate: str | None = "linear", inplace: bool = True
    ) -> dict[str, np.ndarray]:
        """Read the HDF5 file and return `{P, Q, V}` matrices of shape (M timesteps, N nodes)."""
        raw_data = self.__read_raw_mat()
        self.feature_index = np.asarray(list(raw_data["loadNames"][self.nodes])).flatten()
        data = {}
        data["P"] = np.transpose(np.asarray(raw_data["AMI_PkW"]))
        data["Q"] = np.transpose(np.asarray(raw_data["AMI_QkVAR"]))
        data["V"] = np.transpose(np.asarray(raw_data["AMI_Vpu"]))
        hc_baseline = {
            "all": raw_data["HC_Vconstrained"][:, 0],
            "day": raw_data["HC_Vconstrained"][:, 1],
        }
        if inplace:
            self.data = data
            self.hc_baseline = hc_baseline
            if interpolate is not None:
                warnings.warn("Interpolating in place with default method: " + interpolate)
                self.interpolate()
            return self.data
        else:
            return data

    def get_daytime_datasets(self, inplace: bool = False) -> dict[str, np.ndarray]:
        """Restrict the loaded P/Q/V matrices to the 9:00-15:59 (9am through 3pm hour) window."""
        if self.nodal_ami_dfs is None:
            df0 = self.get_nodal_ami_dfs()[0]
        else:
            df0 = self.nodal_ami_dfs[0]
        self.daytime_mask = (df0.index.hour >= 9) & (df0.index.hour <= 15)
        return {
            "V": self.data["V"][self.daytime_mask, :],
            "P": self.data["P"][self.daytime_mask, :],
            "Q": self.data["Q"][self.daytime_mask, :],
        }

    def get_nodal_ami_dfs(self, inplace=False):
        """Make pandas dataframes of (P, Q, V) per node."""
        nodal_ami_dfs = []
        for P_i, Q_i, V_i in zip(self.data["P"].T, self.data["Q"].T, self.data["V"].T):
            D_i = np.vstack([P_i, Q_i, V_i]).T
            nodal_ami_dfs.append(
                pd.DataFrame(data=D_i, index=self.datetime_index, columns=["P", "Q", "V"])
            )
        if inplace:
            self.nodal_ami_dfs = nodal_ami_dfs
        else:
            return nodal_ami_dfs

    def interpolate(self, replace_data_matrices=True, method="linear"):
        """Interpolate AMI data to remove missing data."""
        if self.nodal_ami_dfs is None:
            nodal_ami_dfs = self.get_nodal_ami_dfs()
        else:
            nodal_ami_dfs = self.nodal_ami_dfs

        for i, df in tqdm(enumerate(nodal_ami_dfs), desc="Interpolating"):
            nodal_ami_dfs[i] = df.interpolate(method=method).iloc[1:]

        if replace_data_matrices and self.data:
            self.data["P"] = np.zeros((self.T_steps - 1, self.N_nodes))
            self.data["Q"] = np.zeros((self.T_steps - 1, self.N_nodes))
            self.data["V"] = np.zeros((self.T_steps - 1, self.N_nodes))

            self.datetime_index = self.datetime_index[1:]

            for i, df in enumerate(nodal_ami_dfs):
                self.data["P"][:, i] = df["P"]
                self.data["Q"][:, i] = df["Q"]
                self.data["V"][:, i] = df["V"]
        return nodal_ami_dfs

    def differentiate(self, granularity=1):
        """Compute finite difference deviations of V/P/Q across the loaded matrices."""
        M_diff = self.data["V"].shape[0]
        assert M_diff == self.data["P"].shape[0]
        assert self.N_nodes == self.data["P"].shape[1]

        DV = np.zeros((M_diff, self.N_nodes))
        DP = np.zeros((M_diff, self.N_nodes))
        DQ = np.zeros((M_diff, self.N_nodes))

        for i, (V_i_T, P_i_T, Q_i_T) in enumerate(
            zip(self.data["V"].T, self.data["P"].T, self.data["Q"].T)
        ):
            DV[:, i] = np.gradient(V_i_T) / granularity
            DP[:, i] = np.gradient(P_i_T) / granularity
            DQ[:, i] = np.gradient(Q_i_T) / granularity

        return {"DV": DV, "DP": DP, "DQ": DQ}

    @staticmethod
    def unbias(M):
        """Simple unbiasing via mean centering."""
        return M - np.mean(M, axis=0)

    @staticmethod
    def static_differentiate(
        voltage_mvts, real_mvts, reactive_mvts, granularity=1, diff_method=np.gradient
    ):
        """Finite difference deviations of V/P/Q matrices passed in directly."""
        M, N = voltage_mvts.shape
        assert M == real_mvts.shape[0]
        assert N == real_mvts.shape[1]

        DV, DP, DQ = [], [], []
        for V_i_T, P_i_T, Q_i_T in zip(voltage_mvts.T, real_mvts.T, reactive_mvts.T):
            DV.append(diff_method(V_i_T) / granularity)
            DP.append(diff_method(P_i_T) / granularity)
            DQ.append(diff_method(Q_i_T) / granularity)

        return {
            "DV": np.asarray(DV).T,
            "DP": np.asarray(DP).T,
            "DQ": np.asarray(DQ).T,
        }

    def __read_raw_mat(self):
        return h5py.File(self._data_path, "r")

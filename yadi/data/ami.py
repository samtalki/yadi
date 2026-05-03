"""AMI dataset workflow: load HDF5 PQV matrices and prepare them for sensitivity studies."""

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
        """Generate a model-free HC input dataset for a set of nodes.

        Args:
            data_path: path to the AMI HDF5 file.
            nodes: list of nodes under study. Defaults to range(1379).
            time_steps: range of time steps to consider. Defaults to range(35040).
        """
        if nodes is None:
            nodes = list(range(1379))
        if time_steps is None:
            time_steps = list(range(35040))

        self.data_path__ = data_path
        self.raw_data__ = None
        self.raw_ground_truth_data__ = None

        self.data: dict[str, np.ndarray] = {}
        self.hc_baseline: dict[str, np.ndarray] = {}
        self.capab_baseline: dict[str, np.ndarray] = {}

        self.nodal_ami_dfs: list[pd.DataFrame] | None = None
        self.daytime_mask = None

        self.nodes = nodes
        self.N_nodes: int | None = None
        self.feature_index: np.ndarray | None = None
        self.__setup_nodes(nodes)

        self.time_steps = time_steps
        self.T_steps = None
        self.datetime_index = None
        self.__setup_times(time_steps)

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
        """Restrict the loaded P/Q/V matrices to the 9am-3pm window."""
        if self.nodal_ami_dfs is None:
            df0 = self.get_nodal_ami_dfs()[0]
        else:
            df0 = self.nodal_ami_dfs[0]
        self.daytime_mask = (df0.index.hour >= 9) & (df0.index.hour <= 15)
        day_data = {}
        day_data["V"] = self.data["V"][self.daytime_mask, :]
        day_data["P"] = self.data["P"][self.daytime_mask, :]
        day_data["Q"] = self.data["Q"][self.daytime_mask, :]
        return day_data

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

        for i, df in tqdm(enumerate(nodal_ami_dfs), desc="Interpolating"):
            nodal_ami_dfs[i] = df.interpolate(method=method).iloc[1:]

        if replace_data_matrices and self.data is not None:
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
        """Compute finite-difference deviations of V/P/Q across the loaded matrices."""
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
        """Finite-difference deviations of V/P/Q matrices passed in directly."""
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
        return h5py.File(self.data_path__, "r")

    def __setup_times(self, time_steps):
        if time_steps is None:
            self.time_steps = list(range(35040))
        else:
            self.time_steps = time_steps
        self.T_steps = len(self.time_steps)
        self.datetime_index = pd.date_range("2020", freq="15min", periods=self.T_steps)

    def __setup_nodes(self, nodes):
        if nodes is None:
            self.nodes = list(range(1379))
        else:
            self.nodes = nodes
        self.N_nodes = len(self.nodes)

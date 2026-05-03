import pandas as pd

import yadi.dss.model as model


class DSS_Monitor(model.DSS_Data):
    def __init__(self, redirects, precompile: bool = True, verbose: bool = True) -> None:
        super().__init__(redirects, verbose=verbose, precompile=precompile)

    def export_monitor(self, monitor_name, verbose=False):
        """Exports a single monitor to a dataframe"""
        err = self.dss.Text.Command(f"export monitors {monitor_name}")
        print(f"Monitor {monitor_name} Export Returned {err}")
        monitor_info = self.dss.utils.monitors_to_dataframe()  # Get the monitor info df
        # Get the monitor info df index for monitor_name
        monitor_index = monitor_info.index.get_loc(
            monitor_info.index[monitor_info.index == monitor_name][0]
        )

        # Make timeseries_df from exported csv
        timeseries_df = pd.read_csv(
            monitor_info["FileName"][monitor_index],
            sep=r"\s*,\s*",
            header=0,
            encoding="ascii",
            engine="python",
        )
        if verbose:
            print("monitor_name is: ", monitor_name)
            print("monitor_idex is: ", monitor_index)
        return timeseries_df

    def __set_monitor(
        self, element_name, element_type, mon_name_prefix, power=True, voltage=True, verbose=False
    ):
        """Sets a monitor on element_name of element_type"""
        if power:  # If power monitor is enabled
            mon_name = mon_name_prefix + "power"
            err1 = self.dss.Text.Command(
                f"New Monitor.{mon_name} "
                f"Element={element_type}.{element_name} "
                "terminal=1 PPolar=no mode=1"
            )
            if verbose:
                print(
                    "Monitor type: ",
                    mon_name_prefix + "power",
                    " placed on",
                    element_type,
                    " name: ",
                    element_name,
                    "with errors: ",
                    err1,
                )

        if voltage:
            mon_name = mon_name_prefix + "voltage"
            err2 = self.dss.Text.Command(
                f"New Monitor.{mon_name} "
                f"Element={element_type}.{element_name} "
                "terminal=1 vipolar=yes mode=0"
            )
            if verbose:
                print(
                    "Monitor type: ",
                    mon_name_prefix + "voltage",
                    " placed on",
                    element_type,
                    " name: ",
                    element_name,
                    "with errors: ",
                    err2,
                )

    def set_monitor_all_lines(self, verbose=False):
        """Sets timeseries power monitors on all lines before solving"""
        lines = self.dss.Lines.AllNames()
        for n, line_name in enumerate(lines):
            mon_name_prefix = "mon_" + str(line_name) + "_"
            self.__set_monitor(
                element_name=line_name,
                element_type="Line",
                mon_name_prefix=mon_name_prefix,
                power=True,
                voltage=True,
                verbose=verbose,
            )

    def set_monitor_all_trafos(self, verbose=False):
        """Sets timeseries power monitors on all trafos before solving"""
        trafos = self.dss.Transformers.AllNames()
        for n, trafo_name in enumerate(trafos):
            mon_name_prefix = "mon_" + str(trafo_name) + "_"
            self.__set_monitor(
                element_name=trafo_name,
                element_type="Transformer",
                mon_name_prefix=mon_name_prefix,
                power=True,
                voltage=True,
                verbose=verbose,
            )

    def set_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        for n, load_name in enumerate(loads):
            mon_name_prefix = "mon_" + str(load_name) + "_"
            self.__set_monitor(
                element_name=load_name,
                element_type="Load",
                mon_name_prefix=mon_name_prefix,
                power=True,
                voltage=True,
                verbose=verbose,
            )

    def get_monitor_all_trafos(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        trafos = self.dss.Transformers.AllNames()
        Ijk_dict = dict()
        Pjk_dict = dict()
        Qjk_dict = dict()
        for n, trafo_name in enumerate(trafos):
            Ijk, Pjk, Qjk = self.__get_monitor_timeseries(name=trafo_name, type="Transformer")

            if (len(Ijk.shape) > 1) and (len(Pjk.shape) > 1) and (len(Qjk.shape) > 1):
                self.dss.Transformers.Name(trafo_name)  # set current trafo as active
                phases = self.dss.CktElement.NumPhases()
                for ph in range(phases):
                    Ijk_dict[trafo_name + f".{ph + 1}"] = Ijk[:, ph]
                    Pjk_dict[trafo_name + f".{ph + 1}"] = Pjk[:, ph]
                    Qjk_dict[trafo_name + f".{ph + 1}"] = Qjk[:, ph]

            else:
                Ijk_dict[trafo_name] = Ijk
                Pjk_dict[trafo_name] = Pjk[:, 0]
                Qjk_dict[trafo_name] = Qjk[:, 0]

        Ijk_profiles = pd.DataFrame.from_dict(Ijk_dict)
        Pjk_profiles = pd.DataFrame.from_dict(Pjk_dict)
        Qjk_profiles = pd.DataFrame.from_dict(Qjk_dict)

        dt_index = pd.date_range(start="1/1/2019", periods=self.simulation_steps, freq="h")
        Ijk_profiles = Ijk_profiles.set_index(dt_index)
        Pjk_profiles = Pjk_profiles.set_index(dt_index)
        Qjk_profiles = Qjk_profiles.set_index(dt_index)

        return Ijk_profiles, Pjk_profiles, Qjk_profiles

    def get_monitor_all_lines(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        lines = self.dss.Lines.AllNames()
        Ijk_dict = dict()
        Pjk_dict = dict()
        Qjk_dict = dict()
        for n, line_name in enumerate(lines):
            Ijk, Pjk, Qjk = self.__get_monitor_timeseries(name=line_name, type="Line")
            if (len(Ijk.shape) > 1) and (len(Pjk.shape) > 1) and (len(Qjk.shape) > 1):
                self.dss.Lines.Name(line_name)  # set current line as active
                phases = self.dss.Lines.Phases()
                for ph in range(phases):
                    Ijk_dict[line_name + f".{ph + 1}"] = Ijk[:, ph]
                    Pjk_dict[line_name + f".{ph + 1}"] = Pjk[:, ph]
                    Qjk_dict[line_name + f".{ph + 1}"] = Qjk[:, ph]
            else:
                Ijk_dict[line_name] = Ijk
                Pjk_dict[line_name] = Pjk[:, 0]
                Qjk_dict[line_name] = Qjk[:, 0]

        Ijk_profiles = pd.DataFrame.from_dict(Ijk_dict)
        Pjk_profiles = pd.DataFrame.from_dict(Pjk_dict)
        Qjk_profiles = pd.DataFrame.from_dict(Qjk_dict)

        dt_index = pd.date_range(start="1/1/2019", periods=self.simulation_steps, freq="h")
        Ijk_profiles = Ijk_profiles.set_index(dt_index)
        Pjk_profiles = Pjk_profiles.set_index(dt_index)
        Qjk_profiles = Qjk_profiles.set_index(dt_index)
        return Ijk_profiles, Pjk_profiles, Qjk_profiles

    def get_monitor_all_loads(self, verbose=False):
        """Sets timeseries power monitors on all loads before solving"""
        loads = self.dss.Loads.AllNames()
        voltage_dict = dict()
        kw_dict = dict()
        kvar_dict = dict()
        for n, load_name in enumerate(loads):
            volts, kws, kvars = self.__get_monitor_timeseries(load_name)
            voltage_dict[load_name] = volts
            kw_dict[load_name] = kws
            kvar_dict[load_name] = kvars

        voltage_profiles = pd.DataFrame.from_dict(voltage_dict)
        kw_profiles = pd.DataFrame.from_dict(kw_dict)
        kvar_profiles = pd.DataFrame.from_dict(kvar_dict)

        dt_index = pd.date_range(start="1/1/2019", periods=self.simulation_steps, freq="h")
        voltage_profiles = voltage_profiles.set_index(dt_index)
        kw_profiles = kw_profiles.set_index(dt_index)
        kvar_profiles = kvar_profiles.set_index(dt_index)

        return voltage_profiles, kw_profiles, kvar_profiles

    def __get_monitor_timeseries(self, name, type="Load"):
        """
        Parameters:
        ---
            dss: the dss object
            element: name of the lement
        """

        if type == "Line" or type == "Transformer":
            Ijk_ts = self.export_monitor_voltage(name, type)
            Pjk_ts, Qjk_ts = self.export_monitor_power(name, type)
            return Ijk_ts, Pjk_ts, Qjk_ts
        else:
            voltage_ts = self.export_monitor_voltage(name, type)
            kw_ts, kvar_ts = self.export_monitor_power(name, type)
            return voltage_ts, kw_ts, kvar_ts

    def export_monitor_voltage(self, name, type):
        """Gets the voltage timeseries for an element that is monitored"""
        monitor_name = "mon_" + name + "_voltage"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        voltage_matrix = (
            self.dss.Monitors.AsMatrix()
        )  # N timesteps x M chanels (t, 0, v1, angle 1, ... I1, angle1, ...)
        if type == "Load":
            return voltage_matrix[:, 2]  # interested in v1 for loads

        elif type == "Line":
            self.dss.Lines.Name(name)  # set current line as active
            phases = self.dss.Lines.Phases()
            if phases == 1:
                # numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 4]  # interested in current magnitudes for the lines
            if phases == 3:
                # numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 8::2]  # interested in current magnitudes for the lines

        elif type == "Transformer":
            self.dss.Transformers.Name(name)  # set current line as active
            # self.dss.Circuit.SetActiveElement(type + "." + name)
            phases = self.dss.CktElement.NumPhases()
            if phases == 1:
                # numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 6]  # interested in current magnitudes for the lines
            if phases == 3:
                # numCols = voltage_matrix.shape[1]
                # print(f"phase:{phases}-cols:{numCols}")
                return voltage_matrix[:, 10::2]  # interested in current magnitudes for the lines

    def export_monitor_power(self, name, type):
        """Gets the active and reactive power timeseries for an element monitored"""
        monitor_name = "mon_" + name + "_power"
        # set the active monitor according to name
        self.dss.Monitors.Name(monitor_name)
        power_matrix = self.dss.Monitors.AsMatrix()  # N timesteps x M chanels (t, 0, P1, Q1, ....)
        if type == "Load":
            return power_matrix[:, 2], power_matrix[:, 3]  # interesteed in P1, Q1 for loads
        elif type == "Line":
            # numCols = power_matrix.shape[1]
            # print(f"cols:{numCols}")
            return power_matrix[:, 2::2], power_matrix[
                :, 3::2
            ]  # interesteed in Pjks and Qjks for the phases
        elif type == "Transformer":
            # numCols = power_matrix.shape[1]
            # print(f"cols:{numCols}")
            return power_matrix[:, 2::2], power_matrix[
                :, 3::2
            ]  # interesteed in Pjks and Qjks for the phases

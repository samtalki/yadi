"""OpenDSS data structure: the base class for the yadi DSS wrapper chain."""

import warnings

import numpy as np
import pandas as pd

from yadi.dss._binding import dss

ELEMENT_CLASSES = {
    "Load": dss.Loads,
    "PV": dss.PVsystems,
    "Generator": dss.Generators,
    "Line": dss.Lines,
    "Xfmr": dss.Transformers,
}
LINE_CLASSES = ["Line", "Xfmr"]


class DSS_Data:
    """Data class for OpenDSS network models."""

    def __init__(
        self,
        redirects: str | list[str],
        verbose: bool = True,
        precompile: bool = True,
    ) -> None:
        """Initialize an OpenDSS network model from one or more redirect files."""
        self.dss = dss

        if not isinstance(redirects, list):
            self.redirects = [redirects]
        else:
            self.redirects = redirects

        self.verbose = verbose
        self.Y_net: np.ndarray | None = None
        self.currents_dict: dict[str, complex] = {}
        self.voltages_dict: dict[str, complex] = {}
        self.powers_dict: dict[str, complex] = {}
        self.line_currents_dict: dict[str, np.ndarray] = {}
        self.xfmer_currents_dict: dict[str, np.ndarray] = {}
        self._node_base_voltages: pd.Series | None = None
        self.num_compilations = 0
        if precompile:
            self.compile_dss()

    def compile_dss(self) -> None:
        """Compile every redirect file registered on this instance and solve."""
        for redirect in self.redirects:
            self.redirect(redirect)
            self.solve()
        self.num_compilations += 1
        # Topology may have changed; force recompute of cached base voltages.
        self._node_base_voltages = None
        if self.verbose:
            print(f"DSS Compiled Circuit: {self.dss.Circuit.Name()}")

    def solve(self) -> None:
        """Run `solve` and raise if OpenDSS reports non-convergence."""
        self.run_command("solve")
        if not self.dss.Solution.Converged():
            raise RuntimeError(
                f"OpenDSS solve did not converge for circuit {self.dss.Circuit.Name()!r}."
            )

    @staticmethod
    def run_command(cmd: str) -> None:
        """Run an OpenDSS text command, printing the status if non-empty."""
        status = dss.Text.Command(cmd)
        if status:
            print(f"DSS Status ({cmd}): {status}")

    def redirect(self, filename: str) -> None:
        """Redirect to a single DSS file."""
        if self.verbose:
            print(f"DSS Running file: {filename}")
        self.run_command(f'redirect "{filename}"')

    ##Get methods
    @staticmethod
    def get_all_elements(element="Load"):
        """Return a dataframe of all OpenDSS elements of the given class."""
        if element in ELEMENT_CLASSES:
            return dss.utils.to_dataframe(ELEMENT_CLASSES[element])
        return dss.utils.class_to_dataframe(
            element, transform_string=lambda x: pd.to_numeric(x, errors="ignore")
        )

    def get_node_base_voltages(self) -> pd.Series:
        """Per-node line-to-neutral base voltage in kV, indexed by YNodeOrder name."""
        if self._node_base_voltages is not None:
            return self._node_base_voltages
        node_base_voltages: dict[str, float] = {}
        buses = self.dss.Circuit.AllBusNames()
        y_node_order = self.dss.Circuit.YNodeOrder()
        for bus in buses:
            self.dss.Circuit.SetActiveBus(bus)
            bus_base_voltage = self.dss.Bus.kVBase()
            for node in self.dss.Bus.Nodes():
                key = f"{bus}.{node}"
                if key not in y_node_order:
                    key = key.upper()
                    if key not in y_node_order:
                        warnings.warn(
                            f"node {key!r} on bus {bus!r} is not in YNodeOrder; skipping."
                        )
                        continue
                node_base_voltages[key] = bus_base_voltage
        self._node_base_voltages = pd.Series(node_base_voltages)
        return self._node_base_voltages

    def get_node_voltages_mag_pu(self):
        """Per-node voltage magnitude in pu at the current solution."""
        nodes = self.dss.Circuit.YNodeOrder()
        volts = np.asarray(self.dss.Circuit.YNodeVArray())
        V = volts[0::2] + 1j * volts[1::2]
        v_bases = self.get_node_base_voltages()

        voltages_dict: dict[str, float] = {}
        for i, node in enumerate(nodes):
            base_kv = v_bases[node]
            if base_kv == 0:
                raise RuntimeError(
                    f"node {node!r} has zero base voltage; circuit not energized in this section?"
                )
            voltages_dict[node] = np.abs(V[i]) / (base_kv * 1000)
        self.voltages_dict = voltages_dict
        return voltages_dict

    def get_node_voltages(self):
        """Per-node complex voltage at the current solution."""
        nodes = self.dss.Circuit.YNodeOrder()
        volts = np.asarray(self.dss.Circuit.YNodeVArray())
        V = volts[0::2] + 1j * volts[1::2]
        voltages_dict = {node: V[i] for i, node in enumerate(nodes)}
        self.voltages_dict = voltages_dict
        return voltages_dict

    def get_bus_voltages_pu(self):
        """Solve and return per-unit voltage magnitudes for all buses."""
        self.solve()
        return self.dss.Circuit.AllBusMagPu()

    def get_node_currents(self):
        """Per-node complex injection current at the current solution."""
        nodes = self.dss.Circuit.YNodeOrder()
        currents = np.asarray(self.dss.Circuit.YCurrents())
        I_phasors = currents[0::2] + 1j * currents[1::2]
        currents_dict = {node: I_phasors[i] for i, node in enumerate(nodes)}
        self.currents_dict = currents_dict
        return currents_dict

    def get_node_complex_powers(self):
        """Per-node complex power injection S = V I* at the current solution."""
        nodes = self.dss.Circuit.YNodeOrder()
        volts = np.asarray(self.dss.Circuit.YNodeVArray())
        currents = np.asarray(self.dss.Circuit.YCurrents())
        V = volts[0::2] + 1j * volts[1::2]
        I_phasors = currents[0::2] + 1j * currents[1::2]
        S = V * np.conjugate(I_phasors)
        powers_dict = {node: S[i] for i, node in enumerate(nodes)}
        self.powers_dict = powers_dict
        return powers_dict

    def get_line_data(self):
        """Per-line dict of bus names, terminal/conductor counts, node order, phases, and ampacities."""
        data_lines = {}
        line = self.dss.Lines.First()
        while line:
            name_line = self.dss.Lines.Name()
            data_lines[name_line] = {
                "BusNames": self.dss.CktElement.BusNames(),
                "NumTerminals": self.dss.CktElement.NumTerminals(),
                "NumConductors": self.dss.CktElement.NumConductors(),
                "NodeOrder": self.dss.CktElement.NodeOrder(),
                "Phases": self.dss.Lines.Phases(),
                "NormAmps": self.dss.Lines.NormAmps(),
                "EmergAmps": self.dss.Lines.EmergAmps(),
            }
            line = self.dss.Lines.Next()
        return data_lines

    def get_line_ratings(self):
        """Per-line dict of normal and emergency ampere ratings."""
        ratings_lines = {}
        line = self.dss.Lines.First()
        while line:
            ratings_lines[self.dss.Lines.Name()] = {
                "NormAmps": self.dss.Lines.NormAmps(),
                "EmergAmps": self.dss.Lines.EmergAmps(),
            }
            line = self.dss.Lines.Next()
        return ratings_lines

    def get_conductor_ratings(self):
        """Single-phase conductor normal-amp ratings as a flat per-conductor vector."""
        nodal_line_limits = []
        line = self.dss.Lines.First()
        while line:
            num_conductors = self.dss.CktElement.NumConductors()
            num_phases = self.dss.CktElement.NumPhases()
            if num_conductors != num_phases:
                warnings.warn(
                    "Conductor count != phase count on this line; using the smaller of the two."
                )
                n_ph = min(num_conductors, num_phases)
            else:
                n_ph = num_phases
            three_ph_norm_amps = self.dss.Lines.NormAmps()
            nodal_line_limits.extend([three_ph_norm_amps] * n_ph)
            line = self.dss.Lines.Next()
        return np.asarray(nodal_line_limits)

    def get_line_currents(self, structure="matrix"):
        r"""Per-line complex from/to currents at the current solution.

        ``structure='matrix'`` returns a 2×n_phases matrix per line (rows = terminals 1, 2;
        columns = phases). ``structure='dict'`` returns a nested {terminal: {phase: I}} mapping.
        Asymmetry I_{n1,n2}^{(\phi)} != I_{n2,n1}^{(\phi)} is intentional.
        """
        if structure not in ("matrix", "dict"):
            raise ValueError(f"structure={structure!r} must be 'matrix' or 'dict'.")

        network_line_currents = {}
        line = self.dss.Lines.First()
        while line:
            line_label = self.dss.Lines.Name()
            r2n = np.asarray(self.dss.CktElement.Currents())
            line_currents = r2n[0::2] + 1j * r2n[1::2]
            num_terminals = self.dss.CktElement.NumTerminals()
            num_phases = self.dss.Lines.Phases()

            if num_terminals < 2:
                raise RuntimeError(f"line {line_label!r} has fewer than 2 terminals; floating?")
            if num_phases == 0:
                raise RuntimeError(f"line {line_label!r} has zero phases.")

            if structure == "matrix":
                f_currents = line_currents[0:num_phases]
                t_currents = line_currents[num_phases:]
                network_line_currents[line_label] = np.vstack((f_currents, t_currents))
            else:
                terminal_currents: dict[str, dict[str, complex]] = {}
                node_order = self.dss.CktElement.NodeOrder()
                for term_idx in range(num_terminals):
                    phase_currents: dict[str, complex] = {}
                    for ph_idx in range(num_phases):
                        phase_number = node_order[ph_idx]
                        phase_label = self.__make_phase_label(phase_number)
                        phase_currents[phase_label] = line_currents[term_idx * num_phases + ph_idx]
                    terminal_currents[str(term_idx + 1)] = phase_currents
                network_line_currents[line_label] = terminal_currents

            line = self.dss.Lines.Next()
        self.line_currents_dict = network_line_currents
        return network_line_currents

    def get_xfmr_currents(self, structure="matrix", include_neutral=True):
        """Per-transformer 2×n_conductors complex from/to current matrix at the current solution.

        ``include_neutral=False`` requires the neutral conductor to be the last entry of NodeOrder.
        ``structure='dict'`` is not implemented and falls back to matrix with a warning.
        """
        if structure == "dict":
            warnings.warn(
                "Dict structure not yet implemented for transformer currents; returning matrix."
            )
            structure = "matrix"
        if structure != "matrix":
            raise ValueError(f"structure={structure!r} must be 'matrix' or 'dict'.")

        network_xfmr_currents = {}
        xfmr = self.dss.Transformers.First()
        while xfmr:
            xfmr_label = self.dss.Transformers.Name()
            r2n = np.asarray(self.dss.CktElement.Currents())
            # NumConductors includes the neutral; NodeOrder phase 0 marks it.
            xfmr_currents = r2n[0::2] + 1j * r2n[1::2]
            num_phases = self.dss.CktElement.NumConductors()

            if include_neutral:
                f_currents = xfmr_currents[0:num_phases]
                t_currents = xfmr_currents[num_phases:]
                network_xfmr_currents[xfmr_label] = np.vstack((f_currents, t_currents))
            else:
                if self.dss.CktElement.NodeOrder()[num_phases - 1] != 0:
                    raise RuntimeError(
                        f"transformer {xfmr_label!r}: neutral conductor is not last in NodeOrder; "
                        "re-order phases before calling with include_neutral=False."
                    )
                f_currents = xfmr_currents[0 : num_phases - 1]
                t_currents = xfmr_currents[num_phases:-1]
                network_xfmr_currents[xfmr_label] = np.vstack((f_currents, t_currents))

            xfmr = self.dss.Transformers.Next()
        self.xfmer_currents_dict = network_xfmr_currents
        return network_xfmr_currents

    def get_xfmr_ratings(self):
        """Per-transformer kVA rating, keyed by transformer name."""
        ratings_xfmrs = {}
        xfmr = self.dss.Transformers.First()
        while xfmr:
            ratings_xfmrs[self.dss.Transformers.Name()] = self.dss.Transformers.kVA()
            xfmr = self.dss.Transformers.Next()
        return ratings_xfmrs

    def get_xfmr_data(self):
        """Per-transformer dict of connection (isDelta), windings, tap bounds, terminal/conductor counts, NodeOrder."""
        data_xfmrs = {}
        xfmr = self.dss.Transformers.First()
        while xfmr:
            name_xfmr = self.dss.Transformers.Name()
            data_xfmrs[name_xfmr] = {
                "isDelta": self.dss.Transformers.IsDelta(),
                "NumWindings": self.dss.Transformers.NumWindings(),
                "MinTap": self.dss.Transformers.MinTap(),
                "MaxTap": self.dss.Transformers.MaxTap(),
                "NumTerminals": self.dss.CktElement.NumTerminals(),
                "NumConductors": self.dss.CktElement.NumConductors(),
                "NodeOrder": self.dss.CktElement.NodeOrder(),
            }
            xfmr = self.dss.Transformers.Next()
        return data_xfmrs

    @staticmethod
    def __make_phase_label(phase_number: int) -> str:
        """Map an OpenDSS phase number {1,2,3} to {'a','b','c'}; 0 (neutral) is invalid here."""
        mapping = {1: "a", 2: "b", 3: "c"}
        if phase_number not in mapping:
            raise ValueError(
                f"phase number {phase_number} not in {{1,2,3}}; "
                "0 is the neutral conductor and has no a/b/c label."
            )
        return mapping[phase_number]

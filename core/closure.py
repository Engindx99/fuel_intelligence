# core/closure.py

import numpy as np


class ClosureTracker:

    def __init__(self):

        self.initial_mass = None
        self.initial_energy = None

    def compute_total_mass(self, state):

        return np.sum(

            state.CaCO3
            + state.CaO
            + state.SiO2
            + state.Al2O3
            + state.Fe2O3
            + state.C2S
            + state.C3S
            + state.C3A
            + state.C4AF
            + state.CO2
        )

    def compute_total_energy(self, state, config):

        cp_gas = float(
            config["gas"].get("cp", 1100.0)
        )

        cp_solid = float(
            config["material"].get("cp", 900.0)
        )

        gas_energy = np.sum(
            cp_gas * state.Tg
        )

        solid_energy = np.sum(
            cp_solid * state.Ts
        )

        return gas_energy + solid_energy

    def initialize(self, state, config):

        self.initial_mass = self.compute_total_mass(state)

        self.initial_energy = self.compute_total_energy(
            state,
            config
        )

    def check(self, state, config):

        current_mass = self.compute_total_mass(state)

        current_energy = self.compute_total_energy(
            state,
            config
        )

        mass_error = (
            current_mass - self.initial_mass
        ) / max(self.initial_mass, 1e-12)

        energy_error = (
            current_energy - self.initial_energy
        ) / max(self.initial_energy, 1e-12)

        return {

            "mass_error": mass_error,

            "energy_error": energy_error,

            "mass_total": current_mass,

            "energy_total": current_energy
        }
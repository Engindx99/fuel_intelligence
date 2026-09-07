import numpy as np
from dataclasses import fields

from chemistry.phases import SolidPhases, GasPhases
from chemistry.composition import RAW_MEAL_COMPOSITION


class MassTransport:

    # ======================================================
    # SHIFT SOLID: 0 -> N-1
    # ======================================================
    def shift_solid(self, array, fraction):

        moved = array * fraction

        new = array.copy()

        new[1:] += moved[:-1]
        new[:-1] -= moved[:-1]

        return new

    # ======================================================
    # SHIFT GAS: N-1 -> 0
    # ======================================================
    def shift_gas(self, array, fraction):

        moved = array * fraction

        new = array.copy()

        new[:-1] += moved[1:]
        new[1:] -= moved[1:]

        return new

    # ======================================================
    # MOVE SOLID MATERIAL INSIDE ONE ZONE
    # ======================================================
    def move_solid(self, material, fraction):

        for f in fields(SolidPhases):

            values = getattr(material.solids, f.name)

            setattr(
                material.solids,
                f.name,
                self.shift_solid(values, fraction),
            )

    # ======================================================
    # MOVE KILN PROCESS GAS INSIDE ONE ZONE
    # ======================================================
    def move_process_gas(self, material, fraction):

        for f in fields(GasPhases):

            values = getattr(material.gases, f.name)

            setattr(
                material.gases,
                f.name,
                self.shift_gas(values, fraction),
            )

    # ======================================================
    # MOVE MATERIAL INSIDE ALL ZONES
    # ======================================================
    def move_inside_all_zones(
        self,
        state,
        residence_time,
        dt,
    ):

        fraction = np.clip(
            dt / max(residence_time, 1e-9),
            0.0,
            1.0,
        )

        # --------------------------------------------------
        # SOLID
        # --------------------------------------------------
        for material in state.materials.values():

            self.move_solid(
                material,
                fraction,
            )

        # --------------------------------------------------
        # KILN PROCESS GAS
        #
        # Cooler gas is intentionally excluded.
        # Cooler gas is cooling air and will be handled
        # separately when its flow architecture is defined.
        # --------------------------------------------------
        for zone_name in (
            "preheater",
            "calciner",
            "transition",
            "burning",
        ):

            self.move_process_gas(
                state.materials[zone_name],
                fraction,
            )

    # ======================================================
    # TRANSFER SOLID BETWEEN TWO ZONES
    #
    # src[-1] -> dst[0]
    # ======================================================
    def transfer_solid_zone(
        self,
        source,
        destination,
        fraction,
    ):

        fraction = np.clip(
            fraction,
            0.0,
            1.0,
        )

        for f in fields(SolidPhases):

            src = getattr(source.solids, f.name)
            dst = getattr(destination.solids, f.name)

            moved = src[-1] * fraction

            dst[0] += moved
            src[-1] -= moved

    # ======================================================
    # TRANSFER KILN GAS BETWEEN TWO ZONES
    #
    # src[0] -> dst[-1]
    # ======================================================
    def transfer_gas_zone(
        self,
        source,
        destination,
        fraction,
    ):

        fraction = np.clip(
            fraction,
            0.0,
            1.0,
        )

        for f in fields(GasPhases):

            src = getattr(source.gases, f.name)
            dst = getattr(destination.gases, f.name)

            moved = src[0] * fraction

            dst[-1] += moved
            src[0] -= moved

    # ======================================================
    # TRANSFER SOLID BETWEEN ZONES
    # ======================================================
    def transfer_solid_between_zones(
        self,
        state,
        fraction,
    ):

        self.transfer_solid_zone(
            state.materials["preheater"],
            state.materials["calciner"],
            fraction,
        )

        self.transfer_solid_zone(
            state.materials["calciner"],
            state.materials["transition"],
            fraction,
        )

        self.transfer_solid_zone(
            state.materials["transition"],
            state.materials["burning"],
            fraction,
        )

        self.transfer_solid_zone(
            state.materials["burning"],
            state.materials["cooler"],
            fraction,
        )

    # ======================================================
    # TRANSFER KILN GAS BETWEEN ZONES
    #
    # Burning -> Transition -> Calciner -> Preheater
    # ======================================================
    def transfer_gas_between_zones(
        self,
        state,
        fraction,
    ):

        self.transfer_gas_zone(
            state.materials["burning"],
            state.materials["transition"],
            fraction,
        )

        self.transfer_gas_zone(
            state.materials["transition"],
            state.materials["calciner"],
            fraction,
        )

        self.transfer_gas_zone(
            state.materials["calciner"],
            state.materials["preheater"],
            fraction,
        )

    # ======================================================
    # FEED RAW MEAL
    #
    # External solid feed enters Preheater[0]
    # ======================================================
    def feed_raw_meal(self, state, dt):

        feed = (
            state.Feed_rate
            * dt
        )

        state.feed_mass_in_step = (
            feed
        )

        state.feed_mass_in_rate = (
            state.Feed_rate
        )

        solids = (
            state.materials[
                "preheater"
            ].solids
        )

        for f in fields(SolidPhases):

            if f.name in RAW_MEAL_COMPOSITION:

                solids_array = getattr(solids, f.name)

                solids_array[0] += (
                    feed
                    * RAW_MEAL_COMPOSITION[f.name]
                    / 100000.0
                )

    # ======================================================
    # DISCHARGE CLINKER
    #
    # Cooler[-1] -> clinker outlet
    # ======================================================
    def discharge_clinker(self, state, fraction):

        solids = state.materials["cooler"].solids

        clinker_mass = 0.0

        for f in fields(SolidPhases):

            values = getattr(solids, f.name)

            moved = max(float(values[-1]), 0.0) * fraction

            clinker_mass += moved

            values[-1] -= moved

        state.clinker_mass_out_step = clinker_mass

        state.clinker_mass_out_rate = (
            clinker_mass / max(state.dt, 1.0e-12)
        )

        return state

    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state, dt):

        state.dt = dt

        fraction = np.clip(
            dt
            / max(
                state.residence_time,
                1.0e-9,
            ),
            0.0,
            1.0,
        )

        # ======================================================
        # INTERNAL TRANSPORT
        # ======================================================

        self.move_inside_all_zones(
            state,
            state.residence_time,
            dt,
        )

        # ======================================================
        # ZONE TRANSFER
        # ======================================================

        self.transfer_solid_between_zones(
            state,
            fraction,
        )

        self.transfer_gas_between_zones(
            state,
            fraction,
        )

        # ======================================================
        # FEED
        # ======================================================

        self.feed_raw_meal(
            state,
            dt,
        )

        # ======================================================
        # CLINKER DISCHARGE
        # ======================================================

        self.discharge_clinker(
            state,
            fraction,
        )

        return state
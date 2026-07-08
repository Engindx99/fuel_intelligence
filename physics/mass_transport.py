import numpy as np
from dataclasses import fields

from chemistry.phases import SolidPhases, GasPhases
from chemistry.composition import RAW_MEAL_COMPOSITION


class MassTransport:

    # ======================================================
    # SHIFT ONE PHASE
    # ======================================================
    def shift(self, array, fraction):

        moved = array * fraction

        new = array.copy()

        new[1:] += moved[:-1]
        new[:-1] -= moved[:-1]

        return new


    # ======================================================
    # MOVE ONE MATERIAL
    # ======================================================
    def move_material(self, material, fraction):

        # ---------------- SOLIDS ----------------
        for f in fields(SolidPhases):

            values = getattr(material.solids, f.name)

            setattr(
                material.solids,
                f.name,
                self.shift(values, fraction),
            )

        # ---------------- GASES ----------------
        for f in fields(GasPhases):

            values = getattr(material.gases, f.name)

            setattr(
                material.gases,
                f.name,
                self.shift(values, fraction),
            )


    # ======================================================
    # MOVE INSIDE ALL ZONES
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

        for material in state.materials.values():

            self.move_material(
                material,
                fraction,
            )


    # ======================================================
    # TRANSFER BETWEEN TWO ZONES
    # ======================================================
    def transfer_zone(
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

        # ---------- SOLIDS ----------
        for f in fields(SolidPhases):

            src = getattr(source.solids, f.name)
            dst = getattr(destination.solids, f.name)

            moved = src[-1] * fraction

            dst[0] += moved
            src[-1] -= moved

        # ---------- GASES ----------
        for f in fields(GasPhases):

            src = getattr(source.gases, f.name)
            dst = getattr(destination.gases, f.name)

            moved = src[-1] * fraction

            dst[0] += moved
            src[-1] -= moved


    # ======================================================
    # TRANSFER BETWEEN ZONES
    # ======================================================
    def transfer_between_zones(
        self,
        state,
        fraction,
    ):

        self.transfer_zone(
            state.materials["preheater"],
            state.materials["calciner"],
            fraction,
        )

        self.transfer_zone(
            state.materials["calciner"],
            state.materials["transition"],
            fraction,
        )

        self.transfer_zone(
            state.materials["transition"],
            state.materials["burning"],
            fraction,
        )

        self.transfer_zone(
            state.materials["burning"],
            state.materials["cooler"],
            fraction,
        )


    # ======================================================
    # FEED
    # ======================================================
    def feed_raw_meal(self, state, dt):

        feed = state.Feed_rate * dt

        solids = state.materials["preheater"].solids

        for f in fields(SolidPhases):

            if f.name in RAW_MEAL_COMPOSITION:

                solids_array = getattr(solids, f.name)

                solids_array[0] += (
                    feed
                    * RAW_MEAL_COMPOSITION[f.name]
                    / 100000.0
                )


    # ======================================================
    # DISCHARGE
    # ======================================================
    def discharge_clinker(self, state, fraction,):
    

        solids = state.materials["cooler"].solids

        for f in fields(SolidPhases):

            values = getattr(solids, f.name)

            values[-1] = 0.0

        gases = state.materials["cooler"].gases

        for f in fields(GasPhases):

            values = getattr(gases, f.name)

            values[-1] = 0.0


    # ======================================================
    # APPLY
    # ======================================================
    def apply(self, state, dt):

        fraction = np.clip(
            dt / max(state.residence_time, 1e-9),
            0.0,
            1.0,
        )

        self.move_inside_all_zones(
            state,
            state.residence_time,
            dt,
        )

        self.transfer_between_zones(
            state,
            fraction,
        )

        self.feed_raw_meal(
            state,
            dt,
        )

        self.discharge_clinker(state, fraction)

        return state
from dataclasses import dataclass
import numpy as np

from chemistry.phases import SolidPhases, GasPhases


@dataclass
class ZoneMaterial:
    solids: SolidPhases
    gases: GasPhases


def build_zone_material(N, cell):

    return ZoneMaterial(
        solids=SolidPhases(
            H2O=cell("H2O"),
            Bound_H2O=cell("Bound_H2O"),
            CaCO3=cell("CaCO3"),
            CaO=cell("CaO"),
            SiO2=cell("SiO2"),
            Al2O3=cell("Al2O3"),
            Fe2O3=cell("Fe2O3"),
            C2S=cell("C2S"),
            C3S=cell("C3S"),
            C3A=cell("C3A"),
            C4AF=cell("C4AF"),
        ),
        gases=GasPhases(
            CO2=np.zeros(N, dtype=float),
            H2O=np.zeros(N, dtype=float),
        ),
    )


def get_solid_outlet_composition(zone_material):
    """
    Return the solid-phase composition at the downstream
    outlet cell of a zone.

    This contains phase state only; it does not create or
    remove mass and is not an energy source.
    """
    solids = zone_material.solids

    return {
        "H2O": float(solids.H2O[-1]),
        "Bound_H2O": float(solids.Bound_H2O[-1]),
        "CaCO3": float(solids.CaCO3[-1]),
        "CaO": float(solids.CaO[-1]),
        "SiO2": float(solids.SiO2[-1]),
        "Al2O3": float(solids.Al2O3[-1]),
        "Fe2O3": float(solids.Fe2O3[-1]),
        "C2S": float(solids.C2S[-1]),
        "C3S": float(solids.C3S[-1]),
        "C3A": float(solids.C3A[-1]),
        "C4AF": float(solids.C4AF[-1]),
    }
    
def set_solid_inlet_from_upstream(
    upstream_material,
    downstream_material,
):
    """
    Transfer the upstream solid outlet composition to the
    downstream inlet boundary.

    This is a phase-state handoff only.
    It does not create energy or additional mass.
    """

    upstream = upstream_material.solids
    downstream = downstream_material.solids

    downstream.H2O[0] = upstream.H2O[-1]
    downstream.Bound_H2O[0] = upstream.Bound_H2O[-1]
    downstream.CaCO3[0] = upstream.CaCO3[-1]
    downstream.CaO[0] = upstream.CaO[-1]
    downstream.SiO2[0] = upstream.SiO2[-1]
    downstream.Al2O3[0] = upstream.Al2O3[-1]
    downstream.Fe2O3[0] = upstream.Fe2O3[-1]
    downstream.C2S[0] = upstream.C2S[-1]
    downstream.C3S[0] = upstream.C3S[-1]
    downstream.C3A[0] = upstream.C3A[-1]
    downstream.C4AF[0] = upstream.C4AF[-1]
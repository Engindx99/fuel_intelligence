from chemistry.drying import DryingModel
from chemistry.dehydroxylation import DehydroxylationModel
from chemistry.calcination import CalcinationModel
from chemistry.belite import BeliteModel
from chemistry.alite import AliteModel
from chemistry.c3a import C3AModel
from chemistry.c4af import C4AFModel


class ChemistryModel:

    def __init__(self):

        self.drying = DryingModel()

        self.dehydroxylation = DehydroxylationModel()

        self.calcination = CalcinationModel()

        self.belite = BeliteModel()

        self.alite = AliteModel()

        self.c3a = C3AModel()

        self.c4af = C4AFModel()

    # ======================================================
    # APPLY ALL REACTIONS
    # ======================================================
    def apply(self, state):

        state = self.drying.apply(state)

        state = self.dehydroxylation.apply(state)

        state = self.calcination.apply(state)

        state = self.belite.apply(state)

        state = self.alite.apply(state)

        state = self.c3a.apply(state)

        state = self.c4af.apply(state)

        # ======================================================
        # TOTAL REACTION HEAT SINK
        # ======================================================
        state.Reaction_Q_sink = (
            state.Drying_Q_sink
            + state.Dehydroxylation_Q_sink
            + state.Calcination_Q_sink
            + state.Belite_Q_sink
            + state.Alite_Q_sink
            + state.C3A_Q_sink
            + state.C4AF_Q_sink
        )

        return state
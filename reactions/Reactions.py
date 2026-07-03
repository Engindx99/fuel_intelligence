class ReactionModel:

    def __init__(self, N=5):

        self.N = N

        # ================= UNIVERSAL CONSTANTS =================
        self.R = 8.314462618  # J/(mol*K)
        self.eps = 1e-9

    # ======================================================
    def apply(self, state, dt):

        # ================= TEMPERATURE FIELDS =================
        Tg = state.Tg_burning
        Ts = state.Ts_burning
        Tw = state.Tw_burning

        return state
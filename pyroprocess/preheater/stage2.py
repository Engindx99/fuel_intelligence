from .stage import PreheaterStage


class Stage2(PreheaterStage):

    def __init__(self):
        super().__init__(stage_id=2)
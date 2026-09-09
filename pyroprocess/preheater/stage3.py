from .stage import PreheaterStage


class Stage3(PreheaterStage):

    def __init__(self):
        super().__init__(stage_id=3)
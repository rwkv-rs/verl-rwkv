from verl.utils.tracking import Tracking


class _Backend:
    def __init__(self):
        self.finish_calls = []

    def finish(self, **kwargs):
        self.finish_calls.append(kwargs)


def test_tracking_finish_is_idempotent():
    tracking = Tracking.__new__(Tracking)
    tracking._finished = False
    tracking.logger = {"wandb": _Backend(), "file": _Backend()}

    tracking.finish()
    tracking.finish()

    assert tracking.logger["wandb"].finish_calls == [{"exit_code": 0}]
    assert tracking.logger["file"].finish_calls == [{}]

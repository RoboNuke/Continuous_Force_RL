"""
Mock wandb module for testing.
Simple stub implementation to replace wandb functionality.
"""

class MockRun:
    def __init__(self):
        self.id = "test_run_id"
        self.name = "test_run_name"

    def log(self, data, step=None):
        pass

    def finish(self):
        pass

class MockWandb:
    def __init__(self):
        self.run = None

    def init(self, project=None, entity=None, name=None, group=None, tags=None, config=None, **kwargs):
        self.run = MockRun()
        return self.run

    def log(self, data, step=None):
        pass

    def finish(self):
        if self.run:
            self.run.finish()

# Create global mock instance
_mock_wandb = MockWandb()

# Mock module attributes
run = _mock_wandb.run
init = _mock_wandb.init
log = _mock_wandb.log
finish = _mock_wandb.finish
"""Versioned fixed-budget training protocol, independent of the epoch count."""
from dataclasses import asdict, dataclass
import hashlib
import json

PROTOCOL_ID = 'fixed_budget_v2'

@dataclass(frozen=True)
class TrainingPolicy:
    epochs: int
    protocol_id: str = PROTOCOL_ID
    early_stopping_enabled: bool = False
    validation_every_epochs: int = 1
    best_checkpoint_enabled: bool = True
    lr_scheduler_enabled: bool = True

    def __post_init__(self):
        if (type(self.epochs) is not int or self.epochs < 1
                or self.protocol_id != PROTOCOL_ID
                or self.early_stopping_enabled is not False
                or self.validation_every_epochs != 1
                or self.best_checkpoint_enabled is not True
                or self.lr_scheduler_enabled is not True):
            raise ValueError('Unsupported fixed-budget protocol definition')

    def canonical(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()

    def digest(self):
        return hashlib.sha256(self.canonical()).hexdigest()

    def terminal(self, *, epoch, old_stopped=False):
        if type(epoch) is not int or epoch < 0 or epoch > self.epochs:
            raise ValueError('Epoch out of configured bounds')
        return epoch == self.epochs


def policy(epochs=80):
    """80 is a compatibility default for the existing T04-06 plan, not a limit."""
    return TrainingPolicy(epochs=epochs)

"""T04-06 fixed80_v2 protocol contract (side-effect free).

Never mutate legacy frozen configurations or checkpoint objects.  Integration
into a v2 immutable worker is a separate gated development stage.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class Fixed80Policy:
    protocol_id: str = 'fixed80_v2'
    epochs: int = 80
    early_stopping_enabled: bool = False
    validation_every_epochs: int = 1
    best_checkpoint_enabled: bool = True
    lr_scheduler_enabled: bool = True

    def __post_init__(self):
        if (self.protocol_id != 'fixed80_v2' or self.epochs != 80
                or self.early_stopping_enabled is not False
                or self.validation_every_epochs != 1
                or self.best_checkpoint_enabled is not True
                or self.lr_scheduler_enabled is not True):
            raise ValueError('Unsupported fixed80_v2 protocol definition')

    def canonical(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical()).hexdigest()

    def terminal(self, *, epoch: int, old_stopped: bool = False) -> bool:
        if type(epoch) is not int or epoch < 0 or epoch > self.epochs:
            raise ValueError('Epoch out of fixed80_v2 bounds')
        # old_stopped belongs to the parent protocol; do not interpret it as a
        # new terminal condition. Old checkpoint values may be inherited, but
        # old *runs* cannot be mutated in place.
        return epoch == self.epochs


def policy() -> Fixed80Policy:
    return Fixed80Policy()

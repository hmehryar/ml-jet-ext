"""Explicit protocol-specific stopping decisions; legacy remains the default."""
from train_utils.t04_training_policy import PROTOCOL_ID, policy
LEGACY = 'early_stop_v1'
FIXED = PROTOCOL_ID


def validate_protocol(protocol_id, protocol_hash=None, epochs=None):
    if protocol_id == LEGACY:
        if protocol_hash is not None:
            raise ValueError('Legacy protocol must not carry a v2 protocol hash')
        return
    if protocol_id != FIXED:
        raise ValueError('Unknown training protocol: %r' % (protocol_id,))
    if protocol_hash != policy(epochs).digest():
        raise ValueError('fixed_budget_v2 requires matching immutable protocol hash and epochs')


def may_resume(*, protocol_id, protocol_hash=None, epochs=None, stopped):
    validate_protocol(protocol_id, protocol_hash, epochs)
    return not (protocol_id == LEGACY and stopped)


def should_stop(*, protocol_id, protocol_hash=None, epochs=None, epoch, min_epochs, stagnant, patience):
    validate_protocol(protocol_id, protocol_hash, epochs)
    if protocol_id == FIXED:
        if epoch > epochs:
            raise ValueError('Epoch exceeds configured limit')
        return False
    return epoch >= min_epochs and stagnant >= patience

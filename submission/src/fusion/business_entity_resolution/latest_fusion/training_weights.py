'training-only weights for a macro-owner metric and sparse-address examples'
import numpy as np


def sample_weights(frame, mode='uniform', missing_address_multiplier=4.0):
    if mode == 'uniform':
        return None
    if mode not in ('owner_balanced', 'owner_missing_address'):
        raise ValueError(f'Unknown training weight mode: {mode}')
    if not frame.height:
        raise ValueError('Empty weighted training population')
    _, inverse, count = np.unique(frame['qid'].to_numpy(), return_inverse=True, return_counts=True)
    weights = 1.0/count[inverse]
    if mode == 'owner_missing_address':
        if not np.isfinite(missing_address_multiplier) or missing_address_multiplier < 1:
            raise ValueError('Missing-address multiplier must be finite and >= 1')


        weights *= np.where(frame['target_address_empty'].to_numpy() > 0,
                            missing_address_multiplier, 1.0)
    weights /= weights.mean()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid training weights')
    return weights.astype(np.float32)

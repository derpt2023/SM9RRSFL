"""Target-only training views for an isolated alternating attack experiment.

Ordinary/reference batches, scheduled SGD, the distance penalty, target SGD
displacement boosting and server updates remain in the frozen attack adapter.
Only training-derived auxiliary images receive deterministic per-client-call
views. Global evaluation still uses the original unmodified dataset.
"""
from contextlib import contextmanager
import hashlib
import json

import numpy as np

from scripts.experiments.cifar10 import c7_alignment_attack as aligned

VIEW_SALT = 2026102307
TRANSFORMS = {
    'horizontal_flip_v1': dict(pad=0, horizontal_flip_probability=.5),
    'reflect_crop4_flip_v1': dict(pad=4, horizontal_flip_probability=.5),
}


def target_views(torch, features, transform, seed):
    """One view per auxiliary image per client call, reused across target steps."""
    if transform not in TRANSFORMS:
        raise ValueError('Unknown target-view protocol')
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('The local call seed must be a nonnegative integer')
    if features.ndim != 4 or tuple(features.shape[1:]) != (3,32,32) or not features.shape[0]:
        raise ValueError('Target views require nonempty CIFAR NCHW images')
    settings = TRANSFORMS[transform]
    n = features.shape[0]
    rng = np.random.default_rng([int(seed),VIEW_SALT])
    pad = settings['pad']
    # Draw the same fields even for the no-crop candidate; no global RNG state.
    ys = rng.integers(0,2*pad+1,size=n,dtype=np.int64)
    xs = rng.integers(0,2*pad+1,size=n,dtype=np.int64)
    flips = rng.random(n) < settings['horizontal_flip_probability']
    draws = dict(y=ys.tolist(),x=xs.tolist(),flip=flips.tolist())
    if pad:
        padded = torch.nn.functional.pad(features,(pad,pad,pad,pad),mode='reflect')
        count = torch.arange(n,device=features.device)[:,None,None]
        y = torch.as_tensor(ys,device=features.device)[:,None,None]+torch.arange(32,device=features.device)[None,:,None]
        x = torch.as_tensor(xs,device=features.device)[:,None,None]+torch.arange(32,device=features.device)[None,None,:]
        cropped = padded.permute(0,2,3,1)[count,y,x].permute(0,3,1,2)
    else:
        cropped = features
    flip_mask = torch.as_tensor(flips,device=features.device)[:,None,None,None]
    transformed = torch.where(flip_mask,cropped.flip(-1),cropped).contiguous()
    return transformed, dict(protocol=transform,local_call_seed=int(seed),salt=VIEW_SALT,
        count=int(n),flipped=int(flips.sum()),padding_mode='reflect' if pad else 'none',pad=pad,
        draws_sha256=hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest(),
        policy='One view per auxiliary image per client call; shared across that call target steps. No evaluation image is an optimizer input.')


class _TargetContext:
    def __init__(self, original, views):
        self.original, self.x_attack = original, views

    def __getattr__(self, name):
        return getattr(self.original,name)


def run_profile_attack(ctx, global_vector, profile, **kwargs):
    transform = profile.get('target_transform')
    if transform is None:
        # Exact baseline path, with no extra arithmetic or RNG draws.
        return aligned.run_profile_attack(ctx,global_vector,profile,**kwargs)
    if profile.get('optimizer') != 'sgd' or profile.get('normal_optimizer') != 'sgd':
        raise ValueError('This isolated view experiment preserves target and ordinary SGD')
    target_indices = np.asarray(kwargs['target_indices'],dtype=np.int64).reshape(-1)
    torch = ctx.torch
    index = torch.as_tensor(target_indices,dtype=torch.long,device=ctx.device)
    selected = ctx.x_attack.index_select(0,index)
    views,metadata = target_views(torch,selected,transform,kwargs['seed'])
    metadata['original_auxiliary_indices_sha256'] = hashlib.sha256(target_indices.astype('<i8').tobytes()).hexdigest()
    metadata['local_target_metrics_input'] = 'augmented_auxiliary_views; global round metrics still use unmodified auxiliary and independent evaluation images'
    proxy = _TargetContext(ctx,views)
    arguments = dict(kwargs,target_indices=np.arange(len(target_indices),dtype=np.int64))
    try:
        delta,stats,audit = aligned.run_profile_attack(proxy,global_vector,profile,**arguments)
    except BaseException as error:
        if hasattr(error,'audit'):
            error.audit['target_views'] = metadata
        raise
    audit['target_views'] = metadata
    return delta,stats,audit


@contextmanager
def target_view_scope():
    """Patch only this independent worker process; restore on every exit path."""
    from scripts.experiments.cifar10 import attack_stability_attack as original
    previous = original.run_profile_attack
    original.run_profile_attack = run_profile_attack
    try:
        yield
    finally:
        original.run_profile_attack = previous

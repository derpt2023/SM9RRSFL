"""Explicit no-oracle interpretation of Ding et al. (2024), DOI 70375-w.

This is a new, separately identified reproduction, not a rewrite of old TAD
results. See PAPER_CHOICES for ambiguities resolved before any new training.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ast
import inspect
import math

import numpy as np

from sm9rrsfl.ding13_detector import _IsolationTree, _average_path_length
from sm9rrsfl.torch_backend import numpy_singular_values_from_gram


PAPER_CHOICES = dict(
    version="TAD-no-prior-v1", doi="10.1038/s41598-024-70375-w",
    inspected_pdf_sha256="c8e0238114c73f7d9b3b0853e9a723468509659abc27bcb2822b8bf35ddfabf6",
    detector_inputs="Uploaded local model classifier matrices and iteration index only; no attack labels, ratio, onset, source/target class, clean anchor or validation metrics.",
    features="Successive differences of the local full classifier Gram spectra (global+delta), transposed storage gives the same nonzero spectrum.",
    forest=dict(trees=100, sample_size=256, actual_max_depth=10,
                anomaly_score_threshold=0.5, comparison="strict greater than; no ranked quota",
                threshold_origin="Fixed standard isolation-score decision boundary; TAD article does not specify a numeric score threshold."),
    removal=dict(minimum_observed_differences=3, abnormal_fraction_threshold=0.5,
                 comparison="strict greater than", history="all observed differences"),
    removal_interpretation="Follow honesty equation (1) and page8 abnormal behavior exceeding half; correct the contradictory honesty inequality. L=3 is an explicit choice informed by the experimental narrative. Not the alternative consecutive-three rule.",
    weights=dict(initial="1/N", penalty_factor=0.5,
                 recovery="normal observations double weights below 1/N, no cap",
                 redistribution="Keep suspicious weights; equally redistribute residual mass among currently normal survivors, with nonnegative water filling for excess recovery. If no normal survivor exists, normalize surviving weights; no resurrection.",
                 empty_aggregation="Zero aggregate and zero coefficients when all effective weights are zero; retain health failure, never use the legacy uniform fallback."),
    weight_interpretation="Mass-conserving interpretation of page5 prose, not the internally inconsistent literal third line of equation7. Alpha=.5 is a disclosed implementation default, not a verified paper value.",
    limitations="No author code was available in the linked article. Paper assumes IID/noncollusion and reports up to40% poisoning; Non-IID and60/80% are stress tests. Public CIFAR training/attack remain the user's fixed protocol.")


@dataclass(frozen=True)
class TADDecision:
    weights: dict
    outliers: set
    newly_removed: set


def forest_scores(features, *, seed):
    """Original isolation score, fixed .5 threshold, depth10 really means10."""
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Finite 2D trajectory features required")
    n = len(data)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    count = min(256, n)
    rng = np.random.default_rng(seed)
    paths = np.zeros(n, dtype=np.float64)
    for _ in range(100):
        tree = _IsolationTree.build(data[rng.choice(n, count, replace=False)],
                                   depth=0, max_depth=10, rng=rng)
        paths += [tree.path_length(row) for row in data]
    return np.power(2., -paths / (100 * _average_path_length(count)))


def redistribute(weights, normal_ids, surviving_ids):
    """Equal residual allocation, with bounded nonnegative water filling."""
    if not surviving_ids:
        return
    if not normal_ids:
        total = math.fsum(weights[i] for i in surviving_ids)
        if total <= 0:
            raise ValueError("No positive surviving TAD weight")
        for i in surviving_ids:
            weights[i] /= total
        return
    fixed = set(surviving_ids) - set(normal_ids)
    mass = 1. - math.fsum(weights[i] for i in fixed)
    if mass < -1e-12:
        raise ValueError("Suspicious TAD mass exceeds unity")
    pending = list(normal_ids)
    while pending:
        shift = (max(0., mass) - math.fsum(weights[i] for i in pending)) / len(pending)
        clipped = [i for i in pending if weights[i] + shift < 0]
        if not clipped:
            for i in pending:
                weights[i] += shift
            break
        for i in clipped:
            weights[i] = 0.
        pending = [i for i in pending if i not in clipped]


class PriorFreeTAD:
    """The constructor and decision API cannot accept ground-truth priors."""
    def __init__(self, client_ids, *, seed=0, matrix_offset=0, matrix_shape=(1, 1),
                 compute_backend="numpy", device="auto"):
        if not client_ids or len(set(client_ids)) != len(client_ids):
            raise ValueError("Unique nonempty client ids required")
        self.client_ids = tuple(client_ids)
        self.seed, self.matrix_offset, self.matrix_shape = seed, matrix_offset, matrix_shape
        self.compute_backend, self.device = compute_backend, device
        self.weights = {i:1./len(client_ids) for i in client_ids}
        self.previous_singulars, self.counts, self.abnormal_counts = {}, {}, {}
        self.removed = set()
        self.last_round, self.last_audit = 0, {}

    def evaluate_round(self, updates, *, global_params, round_id):
        if round_id <= self.last_round or set(updates)-set(self.client_ids) or set(updates)&self.removed:
            raise ValueError("Invalid TAD trajectory input")
        self.last_round = round_id
        start = self.matrix_offset
        end = start + math.prod(self.matrix_shape)
        def cpu(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        # Only the small uploaded classifier matrix is materialized on CPU.
        current = {}
        for identity in self.client_ids:
            if identity in updates:
                local = (cpu(global_params[start:end]).astype(np.float32)
                         + cpu(updates[identity][start:end]).astype(np.float32))
                matrix = local.reshape(self.matrix_shape)
                current[identity] = numpy_singular_values_from_gram(matrix)
        features = {i:s-self.previous_singulars[i] for i,s in current.items()
                    if i in self.previous_singulars}
        self.previous_singulars.update(current)
        ids = list(features)
        scores = forest_scores([features[i] for i in ids], seed=self.seed+round_id*7919) if ids else []
        # For an all-identical cloud the exact theoretical score is .5. This
        # equality branch prevents summation roundoff from creating anomalies.
        if ids and all(np.array_equal(features[i], features[ids[0]]) for i in ids):
            scores = np.full(len(ids), .5)
        outliers = {i for i,s in zip(ids,scores) if s > .5}
        newly_removed = set()
        for i in ids:
            self.counts[i] = self.counts.get(i,0)+1
            self.abnormal_counts[i] = self.abnormal_counts.get(i,0)+int(i in outliers)
            if i in outliers:
                self.weights[i] *= .5
            elif self.weights[i] < 1./len(self.client_ids):
                self.weights[i] *= 2.
            if self.counts[i] >= 3 and 2*self.abnormal_counts[i] > self.counts[i]:
                self.weights[i] = 0.
                newly_removed.add(i)
        self.removed.update(newly_removed)
        survivors = [i for i in self.client_ids if i not in self.removed]
        normal = [i for i in survivors if i in current and i not in outliers]
        redistribute(self.weights, normal, survivors)
        self.last_audit = dict(version=PAPER_CHOICES['version'], round=round_id,
            scores={i:float(s) for i,s in zip(ids,scores)}, outliers=sorted(outliers),
            newly_removed=sorted(newly_removed), removed=sorted(self.removed),
            weights=dict(self.weights), observations=dict(self.counts),
            abnormal_counts=dict(self.abnormal_counts))
        return TADDecision(dict(self.weights), outliers, newly_removed)


def adapted_function_source(original):
    """Guarded edits of just the legacy TAD call boundary; old file stays frozen."""
    source = inspect.getsource(original)
    edits = [
        ('def run_experiment(', 'def _run_experiment_prior_free_tad('),
        ('ding13_detector = Ding13TrajectoryDetector(', 'ding13_detector = _PriorFreeTAD('),
        ('            contamination=config.malicious_ratio,\n', ''),
        ('                    update_by_client,\n                    malicious_set,\n                    round_id=round_id,',
         '                    update_by_client,\n                    global_params=params,\n                    round_id=round_id,'),
        ('true_positive_revocations += ding13_result.true_positive_removed',
         'true_positive_revocations += len(ding13_result.newly_removed & malicious_set)'),
        ('false_positive_revocations += ding13_result.false_positive_removed',
         'false_positive_revocations += len(ding13_result.newly_removed - malicious_set)'),
    ]
    for old,new in edits:
        if source.count(old) != 1:
            raise ValueError('Frozen TAD call boundary changed: '+old)
        source = source.replace(old,new)
    begin=source.index('            elif config.method == "ding13":')
    end=source.index('            elif config.method == "vert":',begin)
    branch=source[begin:end]
    if branch.count('aggregate = _weighted_fedavg(')!=1 or branch.count('_normalized_client_coefficients(')!=1:
        raise ValueError('Frozen TAD aggregation boundary changed')
    branch=branch.replace('aggregate = _weighted_fedavg(',
        'aggregate = updates[0] * 0 if not any(w > 0 and n > 0 for w,n in zip(weights,update_samples)) else _weighted_fedavg(')
    branch=branch.replace('_normalized_client_coefficients(', '_tad_client_coefficients(')
    source=source[:begin]+branch+source[end:]
    ast.parse(source)
    return source


def tad_client_coefficients(client_ids, weights, *, sample_counts=None):
    values=np.asarray(weights,dtype=np.float64)
    if sample_counts is not None:values=values*np.maximum(sample_counts,0.)
    total=float(np.maximum(values,0.).sum())
    if total<=0:return dict.fromkeys(client_ids,0.)
    from sm9rrsfl.fl import _normalized_client_coefficients
    return _normalized_client_coefficients(client_ids,weights,sample_counts=sample_counts)


@contextmanager
def prior_free_tad_scope():
    from sm9rrsfl import fl
    from scripts.experiments.cifar10 import attack_stability_worker as worker
    original, evaluator = fl.run_experiment, worker.RoundMetricEvaluator
    if any(k in fl.__dict__ for k in ('_PriorFreeTAD','_tad_client_coefficients','_run_experiment_prior_free_tad')):
        raise RuntimeError('Nested TAD adaptation is not allowed')
    source = adapted_function_source(original)
    fl._PriorFreeTAD = PriorFreeTAD
    fl._tad_client_coefficients = tad_client_coefficients
    # Use the same globals so existing attack/observer scopes keep functioning.
    try:
        exec(compile('from __future__ import annotations\n'+source,
                     __file__+'::adapted_TAD_boundary', 'exec'), fl.__dict__)
    except BaseException:
        del fl._PriorFreeTAD
        del fl._tad_client_coefficients
        raise
    fl.run_experiment = fl._run_experiment_prior_free_tad
    class Evaluator(evaluator):
        def evaluate(self, state):
            row = super().evaluate(state)
            if self.config.method == 'ding13':
                detector = state.get('ding13_detector')
                if not isinstance(detector, PriorFreeTAD):
                    raise ValueError('Wrong TAD detector in checkpoint')
                row['tad_no_prior'] = detector.last_audit
            return row
    worker.RoundMetricEvaluator = Evaluator
    try:
        yield
    finally:
        worker.RoundMetricEvaluator = evaluator
        fl.run_experiment = original
        del fl._PriorFreeTAD
        del fl._tad_client_coefficients
        del fl._run_experiment_prior_free_tad

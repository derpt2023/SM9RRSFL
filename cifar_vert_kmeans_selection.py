"""Explicit author-code K-means selector, scoped to the supplemental workers.

Wang et al. TDSC 2026 VI-C3, pp.2756-2757; mylab426/VERT at
cc71a9dff95085219cdc2d6c033c10c13ed3e6e7, defenses.py lines233-247.
Only selection changes. The frozen predictor, projection and history stay intact.
"""
from contextlib import contextmanager
import json
import warnings

import numpy as np

PARAMETERS = dict(n_clusters=2, random_state=42, init='k-means++', n_init=10,
                  max_iter=300, tol=1e-4, algorithm='lloyd', copy_x=True)


def select(scores):
    """Author's ascending score/zero embedding and cluster containing the maximum.

Pin n_init=10, the effective default of Station's sklearn1.2.0. It is not tuned.
No labels, malicious ratio, evaluation data, or target admitted count are inputs.
"""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    values = np.asarray(scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite similarity scores required')
    if np.any(values[:-1] > values[1:]):
        raise ValueError('Ascending similarity scores required')
    if len(values) == 1:
        return [0], [0], 'single_client'
    features = np.column_stack([values, np.zeros_like(values)])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        labels = KMeans(**PARAMETERS).fit_predict(features)
    chosen = np.flatnonzero(labels == labels[-1]).tolist()
    if chosen != list(range(len(values)-len(chosen), len(values))):
        raise ValueError('High similarity cluster must be a contiguous upper interval')
    return chosen, labels.tolist(), 'kmeans'


@contextmanager
def selection_scope(path, task_id):
    from sm9rrsfl.vert import VERTDefense
    original_cluster = VERTDefense._high_similarity_cluster_size
    original_evaluate = VERTDefense.evaluate_round
    latest = {}

    def cluster(self, ranked):
        if self.top_k != 0 or self.malicious_ratio_prior is not None:
            raise ValueError('Supplement requires K-means without ratio prior')
        values = sorted(ranked)
        chosen, labels, mode = select(values)
        latest.update(scores_ascending=values, labels=labels, mode=mode,
                      selected_count=len(chosen))
        return len(chosen)

    def evaluate(self, updates, *, round_id):
        latest.clear()
        result = original_evaluate(self, updates, round_id=round_id)
        if round_id <= 2:
            details = dict(mode='bootstrap', selected_count=len(updates))
        else:
            if not latest or latest['selected_count'] != len(result.selected_clients):
                raise ValueError('K-means selection was bypassed')
            details = dict(latest)
        row = dict(details, round=round_id, task_id=task_id,
                   scores=result.scores, selected=list(result.selected_clients),
                   rejected=list(result.rejected_clients), weights=result.weights,
                   parameters=PARAMETERS, ratio_prior_used=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        return result

    VERTDefense._high_similarity_cluster_size = cluster
    VERTDefense.evaluate_round = evaluate
    try:
        yield
    finally:
        VERTDefense._high_similarity_cluster_size = original_cluster
        VERTDefense.evaluate_round = original_evaluate


def audit(records, task_id):
    if [r['round'] for r in records] != list(range(1,151)):
        raise ValueError('Exactly150 actual VERT decisions required')
    counts = []
    for r in records:
        if r['task_id'] != task_id or r['ratio_prior_used'] or r['parameters'] != PARAMETERS:
            raise ValueError('Selector identity differs')
        ranked = sorted(r['scores'], key=lambda c:(-r['scores'][c],c))
        if r['round'] <= 2:
            expected = sorted(ranked)
            if r['mode'] != 'bootstrap': raise ValueError('Bootstrap differs')
        else:
            vals = sorted(r['scores'].values())
            chosen, labels, mode = select(vals)
            if (vals != r['scores_ascending'] or labels != r['labels'] or mode != r['mode']):
                raise ValueError('Actual K-means decisions do not replay')
            expected = ranked[:len(chosen)]
        if r['selected'] != expected or r['selected_count'] != len(expected):
            raise ValueError('Actual admitted clients differ')
        if set(r['rejected']) != set(ranked)-set(expected):
            raise ValueError('Rejected clients differ')
        if r['weights'] != {c:1/len(expected) for c in expected}:
            raise ValueError('Actual uniform weights differ')
        counts.append(len(expected))
    return dict(rounds_verified=150, bootstrap_rounds=2, kmeans_rounds=148,
                actual_scores_labels_and_selection_replayed=True,
                selected_counts=counts, minimum=min(counts[2:]), maximum=max(counts[2:]))

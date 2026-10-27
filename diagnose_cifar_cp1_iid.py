#!/usr/bin/env python3
"""Read-only CP1 failure chronology; labels are used only for offline diagnosis."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'outputs/cifar_cp1_distribution_validation150_v2'
OUT = ROOT / 'outputs/cifar_cp1_iid_diagnosis_20261027'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(ds):
    """Summarize observed records, not hypothetical post-intervention training."""
    return dict(observations=len(ds), accepted=sum(d.aggregation_accepted for d in ds),
        history_admitted=sum(d.history_admitted for d in ds),
        coefficient_sum=sum(d.aggregation_weight for d in ds),
        clipped=sum(d.clip_factor < 1 for d in ds),
        scores={k:dict(zip(('min', 'median', 'p90', 'max'),
            np.quantile([getattr(d,k) for d in ds], [0,.5,.9,1]).tolist())) if ds else None
            for k in ('novelty_score','anchor_score','signed_score','class_score','norm_score')})


def cell_audit(result, rows):
    cfg = result.config
    if [r['round'] for r in rows] != list(range(151)) or result.stopped_round != 150:
        raise ValueError('Complete 150-round evidence required')
    ds = result.diagnostics
    by_client = defaultdict(list)
    observed = {}
    for d in ds:
        key = d.round, d.client_id
        if key in observed:
            raise ValueError('Duplicate observation')
        observed[key] = d
        by_client[d.client_id].append(d)
        if d.is_malicious != (d.client_id in result.malicious_clients):
            raise ValueError('Diagnostic label mismatch')
        if d.round > cfg.detector_window:
            warning = d.novelty_score > cfg.detector_distance_threshold
            drift = d.cumulative_drift > cfg.detector_drift_threshold
            if d.suspicious != (warning or drift) or d.count_increment != d.suspicious:
                raise ValueError('Recorded detector predicate differs')
            if d.immediate_revocation != (d.novelty_score > cfg.detector_reject_threshold):
                raise ValueError('Recorded strong predicate differs')
        if d.count_after != (min(cfg.suspicion_remove_after, d.count_before+1)
                            if d.count_increment else d.count_before*.5):
            raise ValueError('Recorded evidence-count recurrence differs')
    revocations = {d.client_id:d.round for d in ds if d.revoked}
    for rid in range(1,151):
        expected = {f'client-{i}' for i in range(100)
                    if revocations.get(f'client-{i}',151) >= rid}
        if {cid for r,cid in observed if r == rid} != expected:
            raise ValueError('Missing client observations without prior revocation')
        for mal, key in ((False,'false_positive_revocations'),(True,'true_positive_revocations')):
            count = sum(d.revoked and d.is_malicious == mal and d.round <= rid for d in ds)
            if count != rows[rid][key]:
                raise ValueError('Metric revocation counter differs')
        mass = sum(d.aggregation_weight for d in ds if d.round == rid and d.is_malicious)
        if abs(mass - rows[rid]['malicious_weight_mass']) > 1e-9:
            raise ValueError('Actual malicious coefficient mismatch')
    fp = []
    for d in ds:
        if d.revoked and not d.is_malicious:
            hist = by_client[d.client_id]
            post = [x for x in hist if x.round > cfg.detector_window]
            fp.append(dict(client=d.client_id, round=d.round,
                trigger='strong' if d.immediate_revocation else 'cumulative_count',
                novelty=d.novelty_score, drift=d.cumulative_drift,
                count_before=d.count_before, count_after=d.count_after,
                first_warning=next((x.round for x in post if x.suspicious),None),
                last_history_admission=max((x.round for x in hist if x.history_admitted),default=None),
                flagged_rounds=sum(x.suspicious for x in post), observed_rounds=len(post)))
    intervals={}
    for lo,hi in ((21,24),(25,25),(26,30),(31,60),(61,84),(85,150),(141,150)):
        intervals[f'{lo}-{hi}']={name:describe([d for d in ds if lo<=d.round<=hi and d.is_malicious==mal])
            for mal,name in ((False,'honest'),(True,'malicious'))}
    # Scores here come from CP1 trajectories; thresholds do NOT predict new trajectories.
    first=[d for d in ds if d.round == 25]
    thresholds={str(t):{name:sum(d.novelty_score>t for d in first if d.is_malicious==mal)
        for mal,name in ((False,'honest_flagged'),(True,'malicious_flagged'))}
        for t in (1.,.9,.85,.8,.75)}
    return dict(tail={k:sum(r[k] for r in rows[-10:])/10 for k in
        ('accuracy','source_transfer_asr','honest_weight_loss','malicious_weight_mass')},
        attack_period_mean={k:sum(r[k] for r in rows[25:])/126 for k in ('accuracy','source_transfer_asr')},
        peak_attack_asr=max(r['source_transfer_asr'] for r in rows[25:]),
        first_attack_period_round_asr_below_20=next((r['round'] for r in rows[25:] if r['source_transfer_asr']<.2),None),
        final_fp=rows[-1]['false_positive_revocations'],
        final_tp=rows[-1]['true_positive_revocations'],
        first_asr_over_80=next((r['round'] for r in rows if r['round']>=25 and r['source_transfer_asr']>.8),None),
        first_malicious_history=next((d.round for d in ds if d.round>=25 and d.is_malicious and d.history_admitted),None),
        false_revocations=fp,intervals=intervals,first_attack_fixed_score_flags=thresholds,
        landmarks=[{k:r[k] for k in ('round','accuracy','source_transfer_asr','malicious_weight_mass',
            'false_positive_revocations','true_positive_revocations')} for r in rows if
            r['round'] in (24,25,26,27,30,40,60,84,85,90,150)],
        actual_rule_count_coverage_and_mass_verified=True)


def main():
    manifest=json.loads((PARENT/'manifest.json').read_text())
    inputs={}
    for field in ('sources','references'):
        for name,digest in manifest[field].items():
            if sha(ROOT/name)!=digest: raise ValueError('Frozen input changed: '+name)
    def read(path, binary=False):
        path=Path(path);inputs[str(path.relative_to(ROOT))]=sha(path)
        return pickle.loads(path.read_bytes()) if binary else json.loads(path.read_text())
    protocol=read(ROOT/'configs/cifar10_cp1_distribution_validation150_v2.json')
    cells={}; labels={}
    for name,c in protocol['cells'].items():
        if c['method']!='sm9rrs':continue
        path=ROOT/c['path']; result=read(path/'result.pickle',True)
        if asdict(result.config)!=c['config']:raise ValueError('Result config differs: '+name)
        rows=read(path/'metrics.json')['rounds']
        cells[name]=cell_audit(result,rows)
        labels[name]=result.malicious_clients
    comparisons={}
    for ratio in ('r20','r40','r60','r80'):
        comparisons[ratio]={'same_malicious_identities':labels['iid_'+ratio+'_sm9rrs']==labels['dirichlet_'+ratio+'_sm9rrs']}
    for name,digest in inputs.items():
        if sha(ROOT/name)!=digest:raise ValueError('Input changed during read')
    report=dict(parent_fingerprint=manifest['fingerprint'],cells=cells,
        distribution_comparisons=comparisons,input_sha256=inputs,
        diagnostic_source_sha256=sha(__file__),
        limitations='Observed same-seed trajectories. Class/signed score separation and chronology are evidence, '
            'not an isolated causal effect of partition, history admission, or a new threshold. '
            'Counterfactual score counts are descriptive; fresh training is required.')
    OUT.mkdir(exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('tail','final_fp','final_tp','first_asr_over_80','first_malicious_history')}
        for k,v in cells.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

"""Frozen-choice confirmation on a new validation seed; no test-driven search."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import hfcdr_validation_protocol as p
import hfcdr_validation_selection as selection
from hfcdr_validation_analysis import report
from hfcdr_pilot_report import assess

RULES = p.ROOT/'configs/hfcdr_confirmation_rules_v1.json'
SPEC = p.ROOT/'configs/hfcdr_validation_confirmation_v1.json'
OUTPUT = p.ROOT/'outputs/hfcdr_validation_confirmation_v1'
METHODS = ('hfcdr', 'vert', 'alignins', 'tad', 'krum', 'fedavg')


def rules():
    value = p.read(RULES)
    if (value['seed'] != 2026111703 or value['formal_seeds'] != [2026111801, 2026111802, 2026111803]
            or value['ratios'] != [0., .2, .4, .6, .8]
            or value['reference_ratios'] != [.4, .8] or value['automatic_search_expansion'] is not False):
        raise ValueError('Predeclared confirmation scope changed')
    original = p.read(p.ROOT/'configs/hfcdr_attack_pilot_v1.json')['criteria']
    if any(original[k] != v for k,v in value['criteria'].items()):
        raise ValueError('Original performance thresholds changed')
    for name, expected in value['sources'].items():
        if p.sha(p.ROOT/name) != expected: raise ValueError('Confirmation reader changed: '+name)
    return value


def make_spec(choice, defense_policy, confirmation_rules):
    if not choice['confirmation_allowed']: raise ValueError('No feasible frozen defense choices')
    expected = {(m,d) for m in defense_policy['defense_candidates'] for d in defense_policy['datasets']}
    decisions = {(d['method'],d['dataset']):d for d in choice['decisions']}
    if len(decisions) != len(choice['decisions']) or set(decisions) != expected:
        raise ValueError('Exactly one frozen choice per method/dataset required')
    policy = deepcopy(defense_policy)
    candidates = {}
    for method in METHODS:
        overrides = {}
        if method != 'fedavg':
            for dataset in policy['datasets']:
                wanted = decisions[method,dataset]['selected']
                candidate = next((c for c in policy['defense_candidates'][method] if c['id'] == wanted), None)
                if candidate is None: raise ValueError('Unregistered or missing selected candidate')
                if dataset in candidate['overrides']:
                    overrides[dataset] = deepcopy(candidate['overrides'][dataset])
        candidates[method] = [dict(id='fixed' if method in ('fedavg','tad','krum') else 'frozen', overrides=overrides)]
    policy.update(phase='confirmation', seeds=[confirmation_rules['seed']], confirmation_seeds=[],
        development_seeds=defense_policy['seeds'], formal_seeds=confirmation_rules['formal_seeds'],
        ratios=confirmation_rules['ratios'], defense_candidates=candidates,
        confirmation_rules=deepcopy(confirmation_rules),
        entry_gate=dict(defense_fingerprint=choice['defense_fingerprint'], selection_sha256=p.digest(choice),
                        requires_complete_defense_validation=True, requires_feasible_frozen_choices=True),
        selected_choices={m:{d:decisions[m,d]['selected'] for d in policy['datasets']}
                          for m in defense_policy['defense_candidates']},
        budget_accounting=dict(fresh_runs=312, candidates_per_method=1, additional_search=False))
    policy.pop('selection', None); policy.pop('selection_sources', None)
    return policy


def upstream(attack, defense):
    defense = p.safe_output(defense); p.check(defense)
    final = p.read(defense/'finalization.json')
    if final['status'] != 'complete' or not final.get('confirmation_allowed'):
        raise ValueError('Complete, qualified defense validation required')
    saved = p.read(defense/'selection.json')
    current = selection.select(attack, defense)
    if saved != current or not saved['confirmation_allowed']:
        raise ValueError('Frozen selection evidence changed or failed')
    manifest = p.read(defense/'manifest.json')
    return make_spec(saved, manifest['policy'], rules())


def build(attack, defense):
    """Produce a reviewable configuration for version control; never trains."""
    policy = upstream(attack, defense)
    if SPEC.exists(): raise FileExistsError('Existing confirmation configuration retained')
    p.write(SPEC, policy)
    result = p.plan(SPEC)
    if len(result['tasks']) != 312: raise ValueError('Full 312-task confirmation matrix required')
    return dict(result['audit'], fingerprint=result['fingerprint'], specification=str(SPEC))


def prepare(attack, defense, output, data_root):
    expected = upstream(attack, defense)
    if p.specification(SPEC) != expected: raise ValueError('Committed configuration differs from frozen selection')
    result = p.prepare(output, data_root, SPEC)
    out = p.safe_output(output)
    p.write(out/'entry_gate.json', expected['entry_gate'])
    p.write(out/'provenance/original_results_before.json', p.read(Path(defense)/'provenance/original_results_before.json'))
    return result


def compare(manifest, assessment):
    policy = manifest['policy']; r = policy['confirmation_rules']; criteria = r['criteria']
    summaries = assessment['summaries']
    lookup = {(t['config']['dataset'], t['config']['partition'], t['condition'],
               t['config']['malicious_ratio'], t['config']['method']):summaries.get(t['task_id'])
              for t in manifest['tasks']}
    if policy['seeds'] != [r['seed']]: raise ValueError('Single independent confirmation seed required')
    clean_rows, cells = [], []
    for dataset in policy['datasets']:
        for partition in policy['partitions']:
            clean = lookup.get((dataset,partition,'clean',0.,'hfcdr'))
            avg_clean = lookup.get((dataset,partition,'clean',0.,'fedavg'))
            clean_ready = all(x is not None and x['complete'] and x['tail10'] is not None for x in (clean,avg_clean))
            clean_checks = dict(complete=clean_ready)
            if clean_ready:
                clean_checks.update(ours_healthy=clean['healthy'], fedavg_healthy=avg_clean['healthy'],
                    clean_accuracy_retained=avg_clean['tail10']['accuracy']-clean['tail10']['accuracy'] <= r['clean_max_accuracy_drop_vs_fedavg']+1e-12,
                    clean_background_retained=clean['tail10']['asr']-avg_clean['tail10']['asr'] <= r['clean_max_asr_increase_vs_fedavg']+1e-12)
            clean_rows.append(dict(dataset=dataset,partition=partition,checks=clean_checks,pass_gate=all(clean_checks.values())))
            for attack in policy['attacks']:
                for ratio in policy['ratios'][1:]:
                    cell = dict(dataset=dataset,partition=partition,condition=attack['name'],malicious_ratio=ratio,
                                reference=ratio in r['reference_ratios'])
                    paired = {m:lookup.get((dataset,partition,attack['name'],ratio,m)) for m in METHODS}
                    if not clean_ready or any(x is None or not x['complete'] or x['tail10'] is None for x in paired.values()):
                        cells.append(dict(cell,status='incomplete',pass_gate=False)); continue
                    outcome = assess(paired['hfcdr']['tail10'], clean['tail10'], paired['fedavg']['tail10'],
                        avg_clean['tail10'], [paired[m]['tail10'] for m in METHODS if m != 'hfcdr'], criteria)
                    effective = outcome['checks']['attack_effective']
                    checks = {k:v for k,v in outcome['checks'].items() if k not in ('attack_effective','attack_reduced')}
                    checks.update(ours_healthy=paired['hfcdr']['healthy'] and clean['healthy'],
                        reference_attack_effective=effective if cell['reference'] else True,
                        reduction_when_effective=outcome['checks']['attack_reduced'] if effective else True)
                    passed = all(checks.values())
                    cells.append(dict(cell,values=outcome['values'],checks=checks,
                        fedavg_attack_effective=effective, relative_reduction_applicable=effective,
                        peer_health={m:paired[m]['healthy'] for m in METHODS if m != 'hfcdr'},
                        pass_gate=passed,status='pass' if passed else 'not_passed'))
    qualified = assessment['complete'] and all(c['pass_gate'] for c in clean_rows+cells)
    return dict(schema='hfcdr_confirmation_decision_v1',fingerprint=manifest['fingerprint'],
        expected_tasks=312,completed=assessment['completed'],complete=assessment['complete'],
        clean_comparisons=clean_rows,comparisons=cells,formal_allowed=qualified,
        official_test_used=False,automatic_search_expansion=False,
        qualification='Independent training-validation seed; no parameter adjustment after confirmation. All failures retained.')


def assess_output(output):
    out = p.safe_output(output); p.check(out); manifest = p.read(out/'manifest.json')
    if manifest['policy']['phase'] != 'confirmation' or manifest['policy']['confirmation_rules'] != rules():
        raise ValueError('Frozen confirmation rules differ')
    controller = p.read(out/'controller.json')
    if controller['status'] != 'complete' or controller['active']:
        raise ValueError('Full confirmation execution must finish before freezing a decision')
    from audit_hfcdr_validation import audit
    evidence = audit(out, full=True)
    if not evidence['all_results_complete']:
        raise ValueError('Incomplete confirmation evidence cannot qualify formal training')
    return compare(manifest, report(out, manifest))

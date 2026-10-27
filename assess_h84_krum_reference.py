#!/usr/bin/env python3
"""Separate post-result user amendment; never overwrite the frozen H84 report."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import run_cifar_h84_full_validation as full

DEFAULT = full.base.ROOT / 'configs/cifar10_h84_krum_iid_reference_exception_v1.json'
EXCEPTION = dict(partition='iid', arm='Krum', waived_reason='clean_accuracy_drop',
                 all_other_health_gates_required=True, no_other_method_exemption=True)


def apply_amendment(report, amendment):
    if (amendment['exception'] != EXCEPTION or not amendment['post_result_user_revision']
            or amendment['mechanism_assessment'] != 'supported_observed_selection_concentration_and_local_overfit'):
        raise ValueError('Unexpected reference exception')
    if report.get('status') != 'complete_audited_development':
        raise ValueError('Full audited matrix required; pending methods cannot be excluded')
    strict = full.assess(report['cells'], report['attack_gates'], report['pre_attack_pairing'])
    if strict != report['assessment']:
        raise ValueError('Frozen strict assessment cannot be reproduced')
    result = deepcopy(strict)
    eligibility = result.pop('method_health')
    krum = eligibility['iid']['Krum']
    original_reasons = list(krum['reasons'])
    krum['reasons'] = [r for r in original_reasons if r != 'clean_accuracy_drop']
    krum.update(valid=not krum['reasons'], original_reasons=original_reasons,
                waived_reasons=['clean_accuracy_drop'] if 'clean_accuracy_drop' in original_reasons else [],
                original_clean_accuracy_drop_retained=True)
    result['reference_eligibility'] = eligibility
    for d in full.PARTITIONS:
        eligible = [a for a in full.METHODS if eligibility[d][a]['valid']]
        for label in full.RATIOS:
            name = d + '_' + label
            metrics = strict['scenarios'][name]['raw_metrics']
            comparison = full.rank_comparison(metrics, eligible, eligibility[d]['H84']['valid'], label == 'clean')
            comparison['reference_policy'] = 'original_health_plus_specific_Krum_IID_clean_utility_exception'
            comparison['krum_included_by_exception'] = d == 'iid' and 'Krum' in eligible and bool(krum['waived_reasons'])
            result['scenarios'][name] = comparison
    passed = strict['prerequisites_pass'] and all(c['passed'] for c in result['scenarios'].values())
    result.update(development_criteria_passed=passed, formal_plan_condition_met=passed,
                  original_strict_assessment=strict, user_amendment=amendment,
                  limitations=[full.DESIGN['limitation'],
                    'Post-result user exception, independently recorded; original strict decision retained.',
                    'Only Krum IID clean utility is waived; all other health failures still exclude it.',
                    'Observed selection concentration and local overfit support an explanation; full causal attribution and historical CUDA score replay are not established.'])
    return result


def verify_diagnosis(amendment):
    root = full.base.ROOT
    path = root / amendment['diagnosis_path']
    if full.shared.file_sha(path) != amendment['diagnosis_sha256']:
        raise ValueError('Mechanism evidence changed')
    evidence = full.base.read(path)
    if (evidence['training_performed'] or evidence['official_test_evaluated']
            or evidence['device'] != 'cpu' or not evidence['initial_models_equal']
            or not evidence['dataset_identity_verified']
            or evidence['config_differences'] != {'method': ['krum', 'fedavg']}):
        raise ValueError('Mechanism evidence provenance differs')
    full.verify_files(evidence['input_sha256'])
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amendment', type=Path, default=DEFAULT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    amendment = full.base.read(args.amendment)
    p, _, manifest = full.plan(full.DEFAULT)
    if amendment['frozen_matrix_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Different matrix/platform')
    evidence = verify_diagnosis(amendment)
    if args.verify_only:
        print(json.dumps(dict(evidence_verified=True, input_files=len(evidence['input_sha256']),
                              training_started=False, strict_report_unchanged=True)))
        return
    out = full.base.ROOT / p['output_dir']
    path = out / 'analysis/full_validation.json'
    if full.base.read(out / 'manifest.json') != manifest:
        raise ValueError('Matrix manifest differs')
    report = full.base.read(path)
    if report.get('fingerprint') != manifest['fingerprint']:
        raise ValueError('Incomplete or wrong report')
    refs, _ = full.references(p)
    if full.base.read(out / 'reference_lock.json') != refs:
        raise ValueError('Matrix references changed')
    full.verify_files(report['input_sha256'])
    before = full.shared.file_sha(path)
    result = apply_amendment(report, amendment)
    result['provenance'] = dict(original_report_sha256=before,
        amendment_sha256=full.shared.file_sha(args.amendment), reader_sha256=full.shared.file_sha(__file__),
        diagnosis_sha256=amendment['diagnosis_sha256'], generated_utc=full.base.utc())
    if full.shared.file_sha(path) != before:
        raise ValueError('Strict report changed while reading')
    full.base.write(out / 'analysis/krum_reference_exception_assessment.json', result)
    print(json.dumps(dict(development_criteria_passed=result['development_criteria_passed'],
        krum_iid_eligibility=result['reference_eligibility']['iid']['Krum'],
        failed_scenarios=[n for n, c in result['scenarios'].items() if not c['passed']],
        original_strict_report_preserved=True), indent=2))


if __name__ == '__main__':
    main()

from copy import deepcopy
import json
import unittest
from unittest.mock import patch
import render_cifar_vert_kmeans as v


def fixture():
    cells={};audits={};trajectories={};per_seed={}
    for seed in v.SEEDS:
        for task in v.full.all_tasks():
            name=task['id']+'_s'+str(seed)
            metrics=dict(accuracy=.6,source_transfer_asr=.1,non_source_accuracy=.6,aux128_confidence=.2,honest_weight_loss=0.,malicious_weight_mass=0.)
            c=dict(status='complete',complete_finite=True,completed_round=150,arm=task['arm'],
                config=dict(seed=seed,partition=task['partition'],malicious_ratio=task['ratio'],vert_top_k=20),
                tail=metrics,final=metrics,attack_period=metrics,peak_attack_asr=.1,final_fp=0,
                health=dict(v14_failure_reasons=[],additional_review_reasons=[],complete_observed_health_pass=True),
                timing=dict(runtime_with_crypto_seconds=10.,runtime_without_crypto_seconds=8.,crypto_wall_seconds=2.,
                    available_and_consistent=True,stages={k:.5 for k in ('crypto_setup_wall_seconds','crypto_packet_wall_seconds','crypto_audit_wall_seconds','crypto_finalize_wall_seconds')}))
            cells[name]=c;audits[name]=dict(summary=c,attack=dict(complete_coverage=True,target_views_and_optimizers_verified=True,actual_calls=0,expected_calls=0))
            trajectories[name]=[dict(metrics,round=i,false_positive_revocations=0) for i in range(151)]
        per_seed[str(seed)]=dict(pairing={t['id']:dict(classification_prefix_equal=True,round24_model_equal=True,passed=True,required=t['arm']!='Krum') for t in v.full.all_tasks() if t['ratio']},
            fedavg_attack_gates={t['id']:dict(passed=True) for t in v.full.all_tasks() if t['arm']=='FedAvg' and t['ratio']})
    old=dict(cells=cells,audited_cells=audits,per_seed=per_seed)
    parent=dict(trajectories=trajectories,files={'analysis/formal_results.json':dict(sha256='original')})
    parent_raw=json.dumps(parent).encode();parent_sha=v.sha(parent_raw)
    r=deepcopy(old);r.update(status='complete_audited_vert_kmeans_replacement',posthoc_user_requested_protocol_correction=True,
        no_retuning_or_selection=True,old_vert_in_current_tables=False,fingerprint='new',replacement_lineage={},input_sha256={})
    for name,c in r['cells'].items():
        if c['arm']!='VERT':continue
        c['config']['vert_top_k']=0
        r['audited_cells'][name]['selection']=dict(actual_scores_labels_and_selection_replayed=True,rounds_verified=150,kmeans_rounds=148,selected_counts=[50]*150)
        r['replacement_lineage'][name]=dict(config_changed_fields=['vert_top_k'],same_seed_initial_model_data_targets_malicious_ids=True,unchanged_bootstrap_rounds_match=True)
        r['input_sha256']['new/tasks/'+name+'/metrics.json']='synthetic'
    r['groups']=v.formal_report.grouped_statistics(r['cells'],v.SEEDS)
    for seed in v.SEEDS:
        s=r['per_seed'][str(seed)];selected={t['id']:r['audited_cells'][t['id']+'_s'+str(seed)] for t in v.full.all_tasks()}
        s['strict']=v.full.assess(selected,s['fedavg_attack_gates'],s['pairing'])
        s['with_previous_krum_exception']=v.formal_report.with_krum_exception(s['strict'])
    bundle=dict(schema='h84-vert-kmeans-corrected-export-v1',report=r,manifest=dict(fingerprint='new',sources={str(i):'' for i in range(95)},protocol=dict(output_dir='new',original_export_sha256=parent_sha)),
        trajectories=deepcopy(trajectories),source_verified=True,all_inputs_unchanged=True,original_export_sha256=parent_sha,original_report_sha256='original')
    seal(bundle)
    return bundle,parent_raw,old


def seal(bundle):bundle['combined_report_sha256']=v.sha((json.dumps(bundle['report'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


class RenderTests(unittest.TestCase):
    def test_complete_replacement_and_unchanged_five_methods(self):
        b,raw,old=fixture()
        with patch.object(v.original_render,'validate',return_value=(old,{})):
            self.assertIs(v.validate(b,raw),b['report'])

    def test_missing_replacement_or_top20_or_changed_other_method_rejected(self):
        for mutate in [lambda b:b['report']['replacement_lineage'].pop(next(iter(b['report']['replacement_lineage']))),
                       lambda b:b['report']['cells']['VERT_iid_clean_s2026102801']['config'].update(vert_top_k=20),
                       lambda b:b['trajectories']['FedAvg_iid_clean_s2026102801'][8].update(accuracy=.7),
                       lambda b:b['report']['per_seed']['2026102801']['with_previous_krum_exception']['scenarios']['iid_clean'].update(passed=False)]:
            b,raw,old=fixture();mutate(b);seal(b)
            with patch.object(v.original_render,'validate',return_value=(old,{})):
                with self.assertRaises(ValueError):v.validate(b,raw)

    def test_pinned_report_digest_required(self):
        b,raw,old=fixture();b['report']['fingerprint']='tampered'
        with patch.object(v.original_render,'validate',return_value=(old,{})):
            with self.assertRaises(ValueError):v.validate(b,raw)

    def test_exact_mnist_palette(self):
        from sm9rrsfl.visualization import METHOD_COLORS
        keys=dict(H84='sm9rrs',VERT='vert',AlignIns='alignins',Krum='krum',TAD='ding13',FedAvg='fedavg')
        self.assertEqual(v.COLORS,{k:METHOD_COLORS[a] for k,a in keys.items()})

if __name__=='__main__':unittest.main()

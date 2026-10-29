from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import render_cifar_vert_paper as v
from tests.test_vert_kmeans_render import fixture as old_fixture,seal


def fixture():
    b,raw,old=old_fixture();r=b['report']
    protocol=json.loads(v.selected_run.DEFAULT.read_text())
    protocol.update(output_dir='new',original_export_sha256=b['original_export_sha256'])
    r.update(status='complete_audited_vert_paper_replacement',no_formal_retuning_or_selection=True,
             selected_validation_report_sha256=v.selected_run.VALIDATION_SHA)
    r.pop('no_retuning_or_selection')
    for name,c in r['cells'].items():
        if c['arm']!='VERT':continue
        c['config'].update(v.selected_run.PARAMETERS)
        r['audited_cells'][name]['actual_weights_verified']=True
        r['replacement_lineage'][name].update(config_changed_fields=['vert_top_k','vert_predict_lr'],
            paper_core_adapter=True,selector_only_adapter=False,validation_report_sha256=v.selected_run.VALIDATION_SHA)
    sources=dict(protocol['validation_frozen_sources'])
    sources.update({f'new_{i}':'synthetic' for i in range(5)})
    b.update(schema='h84-vert-paper-selected-export-v1')
    b['manifest'].update(protocol=protocol,sources=sources,validation_report_sha256=v.selected_run.VALIDATION_SHA,
        official_test_used_for_selection=False,prior_test_results_seen=True,formal_results_cannot_change_selected_configuration=True)
    seal(b);return b,raw,old


def events():
    starts=[dict(event='worker_started',task=n,child_pid=i,admission=dict(uuid=f'good-{i}',free_mb=12000)) for i,n in enumerate(('a','b'))]
    return starts+[dict(event='worker_exited',task=n,child_pid=i,gpu_uuid=f'good-{i}',returncode=0,status='complete') for i,n in enumerate(('a','b'))]


class RenderTests(unittest.TestCase):
    def test_complete_current180_and_fixed_selection(self):
        b,raw,old=fixture()
        with patch.object(v.original_render,'validate',return_value=(old,{})):
            self.assertIs(v.validate(b,raw),b['report'])

    def test_missing_old_or_retuned_replacements_rejected(self):
        mutations=[lambda b:b['report']['replacement_lineage'].pop(next(iter(b['report']['replacement_lineage']))),
            lambda b:b['report']['cells']['VERT_iid_clean_s2026102801']['config'].update(vert_top_k=20),
            lambda b:b['report']['cells']['VERT_iid_clean_s2026102801']['config'].update(vert_predict_lr=.003),
            lambda b:b['report']['audited_cells']['VERT_iid_clean_s2026102801'].update(actual_weights_verified=False),
            lambda b:b['report'].update(selected_validation_report_sha256='wrong'),
            lambda b:b['manifest']['protocol'].update(retune_on_formal_test=True),
            lambda b:b['manifest'].update(official_test_used_for_selection=True),
            lambda b:b['trajectories']['FedAvg_iid_clean_s2026102801'][8].update(accuracy=.7),
            lambda b:b['report']['per_seed']['2026102801']['with_previous_krum_exception']['scenarios']['iid_clean'].update(passed=False)]
        for mutate in mutations:
            b,raw,old=fixture();mutate(b);seal(b)
            with patch.object(v.original_render,'validate',return_value=(old,{})):
                with self.assertRaises(ValueError):v.validate(b,raw)

    def test_exact_validation_report_is_mandatory(self):
        with self.assertRaises(ValueError):v.validation_evidence(b'{}')

    def test_attempt_audit_rejects_duplicates_failures_and_gpu_changes(self):
        good=events();self.assertEqual(v.audit_execution_events(good,{'a','b'})['execution_attempts'],2)
        mutations=[lambda e:e.append(deepcopy(e[0])),lambda e:e.pop(),
            lambda e:e[-1].update(returncode=1),lambda e:e[-1].update(gpu_uuid='different'),
            lambda e:e[0]['admission'].update(free_mb=8191),
            lambda e:e[0]['admission'].update(uuid=v.selected_run.QUARANTINE[0]),
            lambda e:e.append(dict(event='dispatch_blocked'))]
        for mutate in mutations:
            e=deepcopy(good);mutate(e)
            with self.assertRaises(ValueError):v.audit_execution_events(e,{'a','b'})

    def test_execution_record_bound_to_exact_export(self):
        b,_,_=fixture();e=dict(status='complete_execution_audit',source_sha256='sha',combined_report_sha256=b['combined_report_sha256'],
            fingerprint=b['manifest']['fingerprint'],execution_attempts=30,scientific_units=30,failed_attempts=0,
            one_attempt_per_task=True,all_workers_exited_successfully=True,input_sha256={'events':'hash'})
        v.validate_execution(e,b,'sha')
        for key,val in [('source_sha256','wrong'),('execution_attempts',31),('failed_attempts',1),('input_sha256',{})]:
            wrong=dict(e);wrong[key]=val
            with self.assertRaises(ValueError):v.validate_execution(wrong,b,'sha')

    def test_exact_mnist_palette_and_paper_labels(self):
        from sm9rrsfl.visualization import METHOD_COLORS
        keys=dict(H84='sm9rrs',VERT='vert',AlignIns='alignins',Krum='krum',TAD='ding13',FedAvg='fedavg')
        self.assertEqual(v.COLORS,{k:METHOD_COLORS[a] for k,a in keys.items()})
        self.assertEqual(v.LABELS['VERT'],'VERT (paper)')

if __name__=='__main__':unittest.main()

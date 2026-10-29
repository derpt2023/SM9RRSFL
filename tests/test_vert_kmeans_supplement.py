from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import run_cifar_vert_kmeans_supplement as run
import cifar_vert_kmeans_selection as selector
import cifar_vert_kmeans_report as report


class SupplementTests(unittest.TestCase):
    def test_matrix_changes_only_selection_and_preserves_all30_cells(self):
        p,specs,manifest=run.plan()
        _,old,_=run.formal.plan()
        old={s['task_id']:s for s in old}
        self.assertEqual(len(specs),30)
        self.assertEqual({s['config']['seed'] for s in specs},set(p['seeds']))
        self.assertEqual(manifest['maximum_rounds'],4500)
        for s in specs:
            expected=deepcopy(old[s['task_id']]);expected['config']['vert_top_k']=0
            self.assertEqual(s,expected)
        self.assertEqual(len(manifest['sources']),len(set(manifest['sources'])))
        self.assertEqual(len(run.formal.plan()[2]['sources']),89)

    def test_refuses_parameter_or_seed_changes(self):
        for field,value in [('seeds',[1,2,3]),('rounds',30),('selection_only',False),
                            ('selector',dict(selector.PARAMETERS,n_init=1))]:
            with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
                p=run.read(run.DEFAULT);p[field]=value
                f=Path(tmp)/'changed.json';f.write_text(json.dumps(p))
                with self.assertRaises(ValueError):run.plan(f)

    def test_author_commented_code_parity_and_dynamic_count(self):
        from sklearn.cluster import KMeans
        cases=[[.1,.12,.8,.9,.95],[.1,.11,.12,.9,.95],[.5]*100,
               [0.1,0.1000000000001,0.1000000000002]]
        rng=np.random.default_rng(12)
        cases += [sorted(rng.uniform(-1,1,100).tolist()) for _ in range(5)]
        counts=[]
        for values in cases:
            features=np.column_stack([values,np.zeros_like(values)])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                official=KMeans(n_clusters=2,random_state=42).fit_predict(features)
            chosen,labels,_=selector.select(values)
            self.assertEqual(labels,official.tolist())
            self.assertEqual(chosen,np.flatnonzero(official==official[-1]).tolist())
            counts.append(len(chosen))
        self.assertEqual(counts[:3],[3,2,100])
        with self.assertRaises(ValueError):selector.select([float('nan')])

    def test_selector_does_not_consume_training_numpy_rng(self):
        np.random.seed(11);before=np.random.get_state()
        selector.select([-.2,-.1,.8,.9])
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);self.assertTrue(np.array_equal(before[1],after[1]))
        self.assertEqual(before[2:],after[2:])

    def test_real_predictor_bootstrap_logging_and_hook_restoration(self):
        from sm9rrsfl.vert import VERTDefense
        original=VERTDefense.evaluate_round
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'decisions.jsonl'
            defense=VERTDefense(['client-'+str(i) for i in range(5)],parameter_size=8,
                projection_dim=4,history_window=3,predict_epochs=1,top_k=0,seed=21)
            with selector.selection_scope(f,'toy'):
                for rid in range(1,4):
                    updates={'client-'+str(i):np.linspace(i,i+rid,8,dtype=np.float32) for i in range(5)}
                    if rid==3:updates['client-4']=-updates['client-4']
                    out=defense.evaluate_round(updates,round_id=rid)
                    aggregate=sum(out.weights[c]*updates[c] for c in out.selected_clients)
                    defense.finalize_round(updates,aggregate,out)
            rows=[json.loads(x) for x in f.read_text().splitlines()]
            self.assertEqual([r['mode'] for r in rows],['bootstrap','bootstrap','kmeans'])
            self.assertEqual([r['selected_count'] for r in rows[:2]],[5,5])
            self.assertIs(VERTDefense.evaluate_round,original)
            with self.assertRaises(RuntimeError),selector.selection_scope(f,'toy'):
                raise RuntimeError('test restoration')
            self.assertIs(VERTDefense.evaluate_round,original)

    def test_full_decision_audit_rejects_changed_or_missing_decision(self):
        values=[.1,.2,.8,.9];chosen,labels,mode=selector.select(values)
        rows=[]
        for rid in range(1,151):
            scores={str(i):v for i,v in enumerate(values)}
            selected=list(scores) if rid<3 else ['3','2']
            rows.append(dict(round=rid,task_id='toy',parameters=selector.PARAMETERS,
                ratio_prior_used=False,scores=scores,selected=selected,
                rejected=[] if rid<3 else ['1','0'],weights={c:1/len(selected) for c in selected},
                mode='bootstrap' if rid<3 else mode,selected_count=len(selected),
                scores_ascending=values,labels=labels))
        self.assertEqual(selector.audit(rows,'toy')['kmeans_rounds'],148)
        with self.assertRaises(ValueError):selector.audit(rows[:-1],'toy')
        changed=deepcopy(rows);changed[30]['selected']=['0','1']
        with self.assertRaises(ValueError):selector.audit(changed,'toy')

    def test_merge_requires_all_new_vert_and_preserves_other_five(self):
        original=dict(cells={},audited_cells={},per_seed={})
        for seed in (1,2,3):
            for task in run.full.all_tasks():
                name=task['id']+'_s'+str(seed)
                c=dict(status='complete',complete_finite=True,arm=task['arm'],
                    config=dict(seed=seed,partition=task['partition'],malicious_ratio=task['ratio'],vert_top_k=20),
                    tail=dict(accuracy=.6,source_transfer_asr=.1),final_fp=0,
                    health=dict(v14_failure_reasons=[],additional_review_reasons=[],complete_observed_health_pass=True),
                    timing=dict(runtime_with_crypto_seconds=10,runtime_without_crypto_seconds=8,crypto_wall_seconds=2))
                original['cells'][name]=c;original['audited_cells'][name]=dict(summary=c)
            pairing={t['id']:dict(passed=True,required=t['arm']!='Krum') for t in run.full.all_tasks() if t['ratio']}
            original['per_seed'][str(seed)]=dict(pairing=pairing,
                fedavg_attack_gates={t['id']:dict(passed=True) for t in run.full.all_tasks() if t['arm']=='FedAvg' and t['ratio']})
        replacements={k:deepcopy(c) for k,c in original['cells'].items() if c['arm']=='VERT'}
        for c in replacements.values():c['config']['vert_top_k']=0;c['tail']['accuracy']=.61
        audits={k:dict(summary=c) for k,c in replacements.items()}
        checks={str(s):{} for s in (1,2,3)}
        merged=report.combine(original,replacements,audits,checks,[1,2,3])
        for name,c in original['cells'].items():
            self.assertEqual(merged['cells'][name],replacements[name] if c['arm']=='VERT' else c)
        self.assertEqual(original['cells']['VERT_iid_clean_s1']['config']['vert_top_k'],20)
        del replacements['VERT_iid_clean_s1']
        with self.assertRaises(ValueError):report.combine(original,replacements,audits,checks,[1,2,3])


from tests import test_cp1_dynamic_gpu as pool_suite
import cifar_vert_kmeans_scheduler as pool


class SupplementSchedulerTests(pool_suite.PoolTests):
    def setUp(self):
        for name,value in (('pool',pool),('base',run)):
            p=patch.object(pool_suite,name,value);p.start();self.addCleanup(p.stop)
        super().setUp()


if __name__=='__main__':unittest.main()

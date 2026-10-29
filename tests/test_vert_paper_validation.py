from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import cifar_vert_paper_core as core
import run_cifar_vert_paper_validation as run


class PaperCoreTests(unittest.TestCase):
    def test_batched_history_matches_author_sum_loop_optimizer_and_scores(self):
        # Multiple clients preserve the shared predictor/Adam state, as in authors.
        sets = [core.networks(13, 5, 123, 'cpu') for _ in range(2)]
        optimizers = [[torch.optim.Adam(net[1].parameters(),lr=.01),
                       torch.optim.Adam([net[2]],lr=.01),torch.optim.Adam([net[3]],lr=.01)]
                      for net in sets]
        rng=np.random.default_rng(7)
        for _client in range(3):
            arrays=[torch.tensor(rng.normal(size=(4,13)),dtype=torch.float32) for _ in range(3)]
            losses=[core.fit_client(*net,opts,*arrays,5,batched=batched)
                    for net,opts,batched in zip(sets,optimizers,(True,False))]
            np.testing.assert_allclose(*losses,atol=2e-6,rtol=2e-6)
            for left,right in zip(list(sets[0][1].parameters())+list(sets[0][2:]),
                                  list(sets[1][1].parameters())+list(sets[1][2:])):
                torch.testing.assert_close(left,right,atol=3e-6,rtol=3e-5)
            with torch.no_grad():
                predicted=[net[1](net[0](net[2]*arrays[0][-1]+net[3]*arrays[1][-1])) for net in sets]
                torch.testing.assert_close(*predicted,atol=3e-6,rtol=3e-5)

    def test_initialization_is_dense_frozen_raw_and_rng_isolated(self):
        torch.manual_seed(4);state=torch.get_rng_state().clone()
        py=random.getstate();np_state=np.random.get_state()
        projector,predictor,a,b=core.networks(13,5,123,'cpu')
        self.assertEqual(tuple(projector.weight.shape),(5,13))
        self.assertFalse(projector.weight.requires_grad)
        self.assertEqual(tuple(a.shape),(13,));self.assertEqual(tuple(b.shape),(13,))
        self.assertEqual([type(x).__name__ for x in predictor],['Linear','ReLU','Linear','ReLU','Linear'])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        other=core.networks(13,5,124,'cpu')[0]
        self.assertFalse(torch.equal(projector.weight,other.weight))

    def test_real_core_keeps_raw_history_sanitizes_and_logs_dynamic_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'decisions.jsonl'
            obj=core.PaperVERT(['client-'+str(i) for i in range(5)],parameter_size=13,
                projection_dim=5,history_window=3,predict_epochs=2,learning_rate=.01,
                compute_backend='torch',device='cpu',seed=31,audit_path=path,task_id='toy')
            rng=np.random.default_rng(42)
            for rid in range(1,6):
                updates={c:rng.normal(size=13).astype(np.float32) for c in obj.client_ids}
                state=torch.get_rng_state().clone()
                result=obj.evaluate_round(updates,round_id=rid)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                agg=sum(result.weights[c]*updates[c] for c in result.selected_clients)
                obj.finalize_round(updates,agg,result)
                for c in result.rejected_clients:np.testing.assert_array_equal(obj._client_history[-1][c],agg)
                self.assertEqual(obj._global_history[-1].shape,(13,))
            self.assertEqual(len(obj._global_history),3)
            records=[json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual([r['selected_count'] for r in records[:2]],[5,5])
            self.assertEqual(records[-1]['coefficient_dimension'],13)
            self.assertTrue(records[-1]['fresh_projector'])
            self.assertEqual(set(records[-1]['client_order']),set(obj.client_ids))
            self.assertEqual(sum(records[-1]['weights'].values()),1.)

    def test_scopes_restore_even_on_failure_and_sparse_checkpoints_keep_metrics_path(self):
        from sm9rrsfl import fl
        from scripts.experiments.cifar10 import attack_stability_worker as worker
        constructor=fl.VERTDefense;writer=worker._write_pickle
        with self.assertRaises(RuntimeError),core.paper_scope(Path('/unused'),'toy'),core.bounded_checkpoints(150):
            self.assertIsNot(fl.VERTDefense,constructor)
            self.assertIsNot(worker._write_pickle,writer)
            raise RuntimeError('restore')
        self.assertIs(fl.VERTDefense,constructor);self.assertIs(worker._write_pickle,writer)
        with patch.object(worker,'_write_pickle') as actual:
            with core.bounded_checkpoints(150):
                for rid in (0,1,23,24,25,150):worker._write_pickle('checkpoint.pickle',dict(completed_round=rid))
                worker._write_pickle('result.pickle',dict(final=True))
            self.assertEqual(actual.call_count,4)


class ValidationTests(unittest.TestCase):
    def test_paper_first_matrix_has_no_test_seeds_and_only_defense_changes(self):
        p,specs,m=run.plan()
        self.assertEqual(len(specs),30);self.assertEqual(m['maximum_rounds'],4500)
        self.assertFalse(m['official_test_used_for_selection'])
        self.assertEqual({s['config']['seed'] for s in specs},{2026102901})
        for part in ('iid','dirichlet'):
            for ratio in (0.,.2,.4,.6,.8):
                paired=[s for s in specs if s['config']['partition']==part and s['config']['malicious_ratio']==ratio]
                current=next(s for s in paired if s['arm']=='CurrentVERT')['config']
                paper=next(s for s in paired if s['arm']=='PaperVERT')['config']
                self.assertEqual(dict(current,vert_predict_lr=.01),paper)
        pp,ps,pm=run.plan(run.ROOT/'configs/cifar10_vert_paper_resource_probe12_v1.json')
        self.assertEqual(len(ps),1);self.assertEqual(pm['maximum_rounds'],12)

    def test_rejects_formal_test_seed_or_scope_and_budget_changes(self):
        original=run.read(run.DEFAULT)
        for key,value in [('seed',2026102801),('rounds',30),('evaluation_scope','official_test'),
                          ('selection_rule',dict(run.RULE,ranking_target=2))]:
            with tempfile.TemporaryDirectory(dir=run.ROOT) as tmp:
                p=dict(original);p[key]=value;path=Path(tmp)/'changed.json';path.write_text(json.dumps(p))
                with self.assertRaises(ValueError):run.plan(path)

    def test_clear_improvement_requires_all_cells_accuracy_and_clean_utility(self):
        _,specs,_=run.plan();cells={}
        for s in specs:
            attack=s['config']['malicious_ratio']>0
            cells[s['task_id']]=dict(complete_finite=True,arm=s['arm'],config=s['config'],
                tail=dict(accuracy=.6 if not attack else (.55 if s['arm']=='PaperVERT' else .4),
                          source_transfer_asr=.05 if not attack else (.2 if s['arm']=='PaperVERT' else .8)))
        report=dict(cells=cells,attack_pairing={str(i):dict(classification_prefix_equal=True,round24_model_equal=True) for i in range(24)},
                    fedavg_attack_gates={str(i):dict(passed=True) for i in range(8)})
        self.assertTrue(run.assess(report)['passed'])
        failed=deepcopy(report);next(iter(failed['cells'].values()))['complete_finite']=False
        self.assertFalse(run.assess(failed)['passed'])
        for c in report['cells'].values():
            if c['arm']=='PaperVERT' and c['config']['malicious_ratio']==0:c['tail']['accuracy']=.5
        self.assertFalse(run.assess(report)['passed'])

    def test_quarantine_filters_both_previously_unstable_devices(self):
        rows=[dict(uuid=u) for u in (*run.QUARANTINE,'GPU-okay')]
        with patch.object(run.base,'gpu_snapshot',return_value=rows):
            self.assertEqual(run.gpu_snapshot(),[dict(uuid='GPU-okay')])


from tests import test_cp1_dynamic_gpu as pool_suite
import cifar_vert_paper_scheduler as pool


class PaperSchedulerTests(pool_suite.PoolTests):
    def setUp(self):
        for name,value in (('pool',pool),('base',run)):
            p=patch.object(pool_suite,name,value);p.start();self.addCleanup(p.stop)
        super().setUp()


if __name__=='__main__':unittest.main()

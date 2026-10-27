import unittest
import inspect
import pickle
from unittest.mock import patch

import numpy as np

import cifar_tad_no_prior as tad


def detector(n=4):
    return tad.PriorFreeTAD([str(i) for i in range(n)], matrix_shape=(2,2), seed=41)


class PriorFreeTADTests(unittest.TestCase):
    def test_empty_tad_weight_does_not_resurrect_updates_via_uniform_fallback(self):
        from sm9rrsfl import fl
        from sm9rrsfl.datasets import make_synthetic_mnist_like
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        states=[]
        def remove_all(self,updates,**kwargs):
            self.removed=set(self.client_ids);self.weights=dict.fromkeys(self.client_ids,0.)
            return tad.TADDecision(dict(self.weights),set(updates),set(updates))
        with tad.prior_free_tad_scope(),patch.object(tad.PriorFreeTAD,'evaluate_round',remove_all):
            config=fl.ExperimentConfig(method='ding13',attack='none',num_clients=5,
                malicious_ratio=0.,rounds=2,local_epochs=1,batch_size=20,
                early_stop=False,compute_backend='numpy',seed=27)
            result=fl.run_experiment(dataset,config,checkpoint_callback=lambda s:states.append(s))
        np.testing.assert_array_equal(states[0]['params'],states[-1]['params'])
        assert len(result.blacklisted_clients)==5
        assert all(r.accepted_updates==0 for r in result.records[1:])
        assert tad.tad_client_coefficients(['a','b'],[0.,0.],sample_counts=[1,3])=={'a':0.,'b':0.}

    def test_identical_updates_never_force_anomaly_quota(self):
        d=detector(10)
        for r in range(1,9):
            result=d.evaluate_round({i:np.ones(4,dtype=np.float32)*r for i in d.client_ids},
                                    global_params=np.ones(4,dtype=np.float32),round_id=r)
            assert not result.outliers and not result.newly_removed
            assert all(abs(w-.1)<1e-12 for w in result.weights.values())


    def test_api_does_not_accept_ground_truth_or_validation(self):
        assert set(inspect.signature(tad.PriorFreeTAD).parameters)=={
            'client_ids','seed','matrix_offset','matrix_shape','compute_backend','device'}
        assert set(inspect.signature(tad.PriorFreeTAD.evaluate_round).parameters)=={
            'self','updates','global_params','round_id'}


    def test_full_local_model_not_delta_is_used(self):
        d=detector()
        updates={i:np.array([.1,0,0,.1],dtype=np.float32) for i in d.client_ids}
        d.evaluate_round(updates,global_params=np.array([10,0,0,1],dtype=np.float32),round_id=1)
        before=d.previous_singulars['0'].copy()
        d.evaluate_round(updates,global_params=np.array([9,0,0,2],dtype=np.float32),round_id=2)
        assert np.max(np.abs(d.previous_singulars['0']-before))>10


    def test_redistribution_matches_mass_conserving_paper_prose(self):
        weights={'a':1/6,'b':1/3,'c':1/3}
        tad.redistribute(weights,['b','c'],list(weights))
        np.testing.assert_allclose(list(weights.values()),[1/6,5/12,5/12])
        weights={'a':.49,'b':.02,'c':.7}
        tad.redistribute(weights,['b','c'],list(weights))
        assert weights['b']==0 and abs(sum(weights.values())-1)<1e-12


    def test_accumulated_majority_not_consecutive_three_and_pickle_resume(self):
        d=detector()
        updates={i:np.arange(4,dtype=np.float32)*(int(i)+1) for i in d.client_ids}
        d.evaluate_round(updates,global_params=np.zeros(4),round_id=1)
        # Client0: abnormal, normal, abnormal. Cumulative2/3, never consecutive3.
        for r,s in [(2,.8),(3,.1),(4,.8)]:
            with patch.object(tad,'forest_scores',return_value=np.array([s,.1,.1,.1])):
                result=d.evaluate_round({i:v*r for i,v in updates.items()},global_params=np.zeros(4),round_id=r)
            if r==3:
                d=pickle.loads(pickle.dumps(d))
        assert result.newly_removed=={'0'} and result.weights['0']==0
        assert abs(sum(result.weights.values())-1)<1e-12


    def test_actual_maximum_tree_depth_not_log_sample_capped(self):
        original=tad._IsolationTree.build
        depths=[]
        def build(*args,**kwargs):
            depths.append(kwargs['max_depth']);return original(*args,**kwargs)
        with patch.object(tad._IsolationTree,'build',side_effect=build):
            tad.forest_scores(np.arange(12).reshape(6,2),seed=1)
        assert set(depths)=={10}


    def test_real_training_boundary_has_no_ratio_or_labels_and_restores(self):
        from sm9rrsfl import fl
        from sm9rrsfl.datasets import make_synthetic_mnist_like
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        original=fl.run_experiment
        captures=[]
        evaluate=tad.PriorFreeTAD.evaluate_round
        def observed(self,*args,**kwargs):
            result=evaluate(self,*args,**kwargs)
            captures.append(pickle.dumps(self.last_audit));return result
        results=[]
        with tad.prior_free_tad_scope(),patch.object(tad.PriorFreeTAD,'evaluate_round',observed):
            for ratio in (0.,.8):
                config=fl.ExperimentConfig(method='ding13',attack='none',num_clients=5,
                    malicious_ratio=ratio,rounds=4,local_epochs=1,batch_size=20,
                    early_stop=False,compute_backend='numpy',seed=27)
                results.append(fl.run_experiment(dataset,config))
        assert fl.run_experiment is original
        assert captures[:4]==captures[4:]
        assert [r.accuracy for r in results[0].records]==[r.accuracy for r in results[1].records]

import copy
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_cifar_c7_station as entry
from sm9rrsfl import fair_tuning
from scripts.experiments.cifar10 import fair_tuning_dashboard as ui


class ControllerTests(unittest.TestCase):
    def test_exact_fixed_matrix_and_display_work(self):
        p, spec, args, tasks = entry.load_plan()
        self.assertEqual(set(tasks), {'validation', 'final'})
        for phase, seeds in [('validation', {301,302,303}), ('final', {401,402,403})]:
            rows = tasks[phase]
            self.assertEqual(len(rows), 180)
            self.assertEqual({r.config.seed for r in rows}, seeds)
            self.assertEqual({r.config.partition for r in rows}, {'iid','dirichlet'})
            self.assertEqual({r.config.malicious_ratio for r in rows}, {0,.2,.4,.6,.8})
            self.assertEqual({r.config.rounds for r in rows}, {150})
            self.assertEqual({r.config.num_clients for r in rows}, {100})
            self.assertEqual({r.config.attack_stealth_steps for r in rows}, {10})
            self.assertEqual(len({entry.runtime_config_key(r.config) for r in rows}), 180)
            for r in rows:
                for key, value in entry.FIXED[r.method].items():
                    self.assertEqual(getattr(r.config, key), value)
        _, _, work = ui.plan(spec)
        self.assertEqual(work['total_tasks'], 360)
        self.assertEqual(work['validation_rounds'] + work['final_rounds'], 54000)
        self.assertTrue(args.resume)
        self.assertEqual(spec.objective_mode, 'fixed_no_selection')

    def test_dry_run_never_inspects_cuda_or_loads_data_or_creates_output(self):
        from scripts.experiments.cifar10 import c7_station_gpu_pool as pool
        with patch.object(entry, 'load_image_dataset') as data, patch.object(entry, 'run') as run, \
                patch.object(pool, 'check_gpus') as check, redirect_stdout(io.StringIO()) as stream:
            entry.main(['--dry-run'])
        data.assert_not_called()
        run.assert_not_called()
        check.assert_not_called()
        self.assertIn('"total_rounds": 54000', stream.getvalue())

    def test_config_is_standalone_and_rejects_protocol_or_output_substitution(self):
        p = json.loads(entry.DEFAULT_CONFIG.read_text())
        self.assertNotIn('reference_dir', p)
        self.assertNotIn('source_protocol', p)
        for key, value in [('output_dir','outputs/cifar10_v14_fair_grid'),
                           ('output_dir','outputs/cifar10_c7_station_150r_v1/nested'),
                           ('rounds',60), ('num_clients',50), ('device','cpu')]:
            q = copy.deepcopy(p)
            q['shared_parameters'][key] = value
            with TemporaryDirectory() as d:
                path = Path(d) / 'bad.json'
                path.write_text(json.dumps(q))
                with self.assertRaises(ValueError):
                    entry.load_plan(path)
        q = copy.deepcopy(p)
        q['attack_profile']['scope'] = 'all'
        with TemporaryDirectory() as d:
            path = Path(d) / 'bad.json'
            path.write_text(json.dumps(q))
            with self.assertRaises(ValueError):
                entry.load_plan(path)

    def test_manifest_contains_full_profile_sources_and_unique_task_directories(self):
        p, spec, _, tasks = entry.load_plan()
        manifest = entry.make_manifest(p, spec, tasks)
        self.assertEqual(len(manifest['task_specs']), 360)
        self.assertEqual(len({t['task_dir'] for t in manifest['task_specs']}), 360)
        self.assertIn('scripts/experiments/cifar10/attack_stability_attack.py', manifest['source_files'])
        self.assertIn('scripts/experiments/cifar10/c7_station_runtime.py', manifest['source_files'])
        self.assertNotIn('sm9rrsfl/_native_rrs.c', manifest['source_files'])
        self.assertEqual(manifest['protocol']['attack_profile'], entry.PROFILE)
        self.assertFalse(manifest['official_test_used_for_selection'])
        self.assertEqual(manifest['fingerprint'], entry.digest({k:v for k,v in manifest.items() if k!='fingerprint'}))

    def test_dashboard_displays_profile_gpu_parameters_and_restores_class(self):
        _, spec, _, tasks = entry.load_plan()
        original = ui.Progress
        @contextmanager
        def stub(spec, **kwargs):
            yield ui.Progress(spec)
        with patch.object(ui, 'dashboard', stub):
            with entry.station_dashboard(spec) as state:
                selected = [next(t for t in tasks['final'] if t.method == m) for m in ('sm9rrs','vert','alignins')]
                rows = [replace(t, config=replace(t.config, device=f'cuda:{i}')) for i,t in enumerate(selected)]
                state.register(rows)
                for t in rows:
                    state.advance(state.start(t.config), 25)
                lines = state.lines(width=180, height=30)
                text = '\n'.join(lines)
                self.assertIn('C7 Adam=.001', text)
                self.assertIn('official-test 10000', text)
                self.assertIn('dim=256', text)
                self.assertIn('sparsity=0.3', text)
                self.assertIn('rounds=75/54000', text)
                for width,height in [(79,23),(39,9),(1,1)]:
                    lines = state.lines(width,height)
                    self.assertLessEqual(len(lines),height)
                    self.assertTrue(all(ui._display_width(line)<=width for line in lines))
        self.assertIs(ui.Progress, original)

    def simulated_run(self, output, *, report_ok=True, fail_phase=None):
        from scripts.experiments.cifar10 import c7_station_runtime as runtime
        from scripts.experiments.cifar10 import c7_station_gpu_pool as pool
        from scripts.experiments.cifar10 import c7_station_report as reporting
        p, spec, args, tasks = entry.load_plan()
        args.output_dir = str(output)
        calls = []
        validation_data, final_data = object(), object()
        @contextmanager
        def context(*args, **kwargs):
            yield SimpleNamespace(gpu_plan=lambda _:None, scheduler_event=lambda *a,**k:None)
        def phase(dataset, rows, args, **kwargs):
            name = rows[0].phase
            calls.append((name, dataset, kwargs['fingerprint_context']))
            if name == fail_phase:
                raise RuntimeError('worker failed')
            results = [(t, SimpleNamespace(stopped_round=150, nonfinite_updates=0)) for t in rows]
            kwargs['on_snapshot'](results, name+'-fingerprint', 'complete')
            return results, name+'-fingerprint'
        report = dict(all_complete=report_ok, all_numerically_finite=True,
                      report_integrity_errors=[] if report_ok else ['missing sidecar'],
                      all_observed_health_pass=False)
        with patch.object(entry, 'make_manifest', return_value={'fingerprint':'test'}), \
                patch.object(entry, 'load_image_dataset'), \
                patch.object(entry, 'stratified_training_three_way_split', return_value=SimpleNamespace(calibration_dataset=validation_data, main_dataset=final_data)), \
                patch.object(entry, 'validate_targeted_split'), patch.object(entry, 'split_metadata', return_value={'train_count':40000}), \
                patch.object(entry, 'station_dashboard', context), patch.object(pool,'gpu_pool',context), \
                patch.object(runtime,'c7_station_runtime',context), \
                patch.object(fair_tuning,'prepare_tuning_tasks',side_effect=lambda d,t,a,**kw:(t,8,'torch:cuda',1)), \
                patch.object(fair_tuning,'execute_resumable_tuning_phase',side_effect=phase), \
                patch.object(entry.experiments,'write_result_files'), \
                patch.object(reporting,'build_report',return_value=report), \
                patch.object(fair_tuning,'select_best_trials',side_effect=AssertionError('must not reselect')):
            entry.run(p,spec,args,tasks,'log')
        self.assertEqual([r[0] for r in calls], ['validation','final'])
        self.assertIs(calls[0][1], validation_data)
        self.assertIs(calls[1][1], final_data)
        self.assertEqual(calls[1][2]['evaluation_scope'], 'official_test')
        self.assertEqual(calls[0][2]['c7_attack_profile'], entry.PROFILE)

    def test_complete_but_unhealthy_comparisons_do_not_skip_official_test(self):
        with TemporaryDirectory() as d:
            self.simulated_run(Path(d))
            state=json.loads((Path(d)/'controller_status.json').read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['completed_tasks'],360)

    def test_missing_evidence_or_worker_failure_never_marks_all_complete(self):
        for kwargs in ({'report_ok':False},{'fail_phase':'validation'}):
            with TemporaryDirectory() as d:
                with self.assertRaises(RuntimeError):
                    self.simulated_run(Path(d),**kwargs)
                state=json.loads((Path(d)/'controller_status.json').read_text())
                self.assertEqual(state['status'],'interrupted_or_failed')

    def test_resume_retains_final_phase_index_if_validation_is_interrupted(self):
        with TemporaryDirectory() as d:
            output = Path(d)
            self.simulated_run(output)
            before = json.loads((output/'controller_status.json').read_text())
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                self.simulated_run(output, fail_phase='validation')
            after = json.loads((output/'controller_status.json').read_text())
            self.assertEqual(after['phases'], before['phases'])
            self.assertEqual(after['phases']['final']['fingerprint'], 'final-fingerprint')
            self.assertEqual(after['status'], 'interrupted_or_failed')

    def test_unidentified_or_changed_output_is_rejected_without_data_load(self):
        for filename, content in [('old.txt','keep'), ('protocol_manifest.json','{}')]:
            with TemporaryDirectory() as d:
                path=Path(d)/filename
                path.write_text(content)
                with self.assertRaises(ValueError):
                    self.simulated_run(Path(d))
                self.assertEqual(path.read_text(),content)


if __name__ == '__main__':
    unittest.main()

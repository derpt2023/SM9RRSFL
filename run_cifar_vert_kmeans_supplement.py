"""Same-seed VERT-only supplement. Never overwrite original scientific records."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import run_cifar_h84_formal as formal
from run_cifar_h84_formal import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory
import run_cifar_h84_full_validation as full
from cifar_vert_kmeans_selection import PARAMETERS, selection_scope

DEFAULT = ROOT/'configs/cifar10_vert_kmeans_supplement150_v1.json'
OWN = ('run_cifar_vert_kmeans_supplement.py','cifar_vert_kmeans_selection.py',
       'cifar_vert_kmeans_scheduler.py','cifar_vert_kmeans_report.py',
       'tests/test_vert_kmeans_supplement.py')
QUARANTINE = 'GPU-5d9f949a-f126-67b3-5c96-af785e33aa5c'


def gpu_snapshot():
    return [g for g in formal.gpu_snapshot() if g['uuid'] != QUARANTINE]


def plan(config=DEFAULT):
    amendment = read(config)
    p, old_specs, old_manifest = formal.plan(ROOT/amendment['parent_config'])
    if (amendment['selector'] != PARAMETERS or amendment['seeds'] != p['seeds']
            or amendment['rounds'] != 150 or amendment['selection_only'] is not True
            or amendment['output_dir'] != 'outputs/cifar_h84_formal_vert_kmeans150_v1'
            or amendment['quarantine_gpu_uuid'] != QUARANTINE):
        raise ValueError('Only the declared same-seed selection correction is allowed')
    specs=[]
    for old in old_specs:
        if old['arm'] != 'VERT': continue
        new=deepcopy(old); new['config']['vert_top_k']=0
        if new['config']['vert_use_ratio_prior'] is not False:
            raise ValueError('No malicious-ratio prior is permitted')
        specs.append(new)
    if len(specs)!=30: raise ValueError('Exactly30 VERT cells required')
    source=dict(old_manifest['sources'])
    names=set(OWN)|{str(Path(config).resolve().relative_to(ROOT)), amendment['parent_config']}
    source.update({n:sha(ROOT/n) for n in names})
    runtime=dict(p, output_dir=amendment['output_dir'], amendment=amendment,
                 parent_output_dir=p['output_dir'])
    identity=dict(protocol=amendment,specs=specs,data_config=old_manifest['data_config'],
        sources=source,parent_fingerprint=old_manifest['fingerprint'],maximum_rounds=4500,
        posthoc_user_requested_protocol_correction=True, test_used_to_tune_parameters=False)
    return runtime,specs,dict(identity,fingerprint=digest(identity))


def references(p, manifest):
    original=ROOT/p['parent_output_dir']
    om=read(original/'manifest.json')
    if om['fingerprint'] != manifest['parent_fingerprint']:
        raise ValueError('Original formal manifest differs')
    lock=deepcopy(read(original/'reference_lock.json'))
    for rel in ('manifest.json','reference_lock.json','analysis/formal_results.json',
                'analysis/recovery_final_audit.json'):
        file=original/rel
        lock['references'][str(file.relative_to(ROOT))]=sha(file)
    if sha(original/'analysis/formal_results.json') != p['amendment']['original_report_sha256']:
        raise ValueError('Original audited report differs')
    report=read(original/'analysis/formal_results.json')
    if report['status']!='complete_audited_formal' or len(report['cells'])!=180:
        raise ValueError('Original complete180 report required')
    full.verify_files(lock['references'])
    parent=read(ROOT/p['development_config'])
    data,contract=formal.official_data(manifest['data_config'],parent['expected_dataset_identity'])
    from scripts.experiments.cifar10 import attack_stability_worker as w
    if w._dataset_identity(data,contract)!=lock['dataset_identity']:
        raise ValueError('Formal training/auxiliary/test data changed')
    lock['selector_environment']=check_selector_environment()
    return lock


def summarize(p,specs,manifest,out):
    from cifar_h84_formal_report import summarize as report
    return report(p,specs,manifest,out)


def check_selector_environment():
    import sklearn
    import scipy
    import numpy
    from sklearn.cluster import _kmeans
    if sklearn.__version__!='1.2.0':
        raise ValueError('Pin scikit-learn1.2.0 to match reviewed Station implementation')
    return dict(sklearn=sklearn.__version__,numpy=numpy.__version__,scipy=scipy.__version__,
                kmeans_source_sha256=sha(Path(_kmeans.__file__)),parameters=PARAMETERS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--max-gpus',type=int,default=8)
    group=parser.add_mutually_exclusive_group()
    for flag in ('batch','ready','summary','finish'):group.add_argument('--'+flag,action='store_true')
    group.add_argument('--task')
    args=parser.parse_args();p,specs,manifest=plan(args.config);out=ROOT/p['output_dir']
    if args.task:
        if read(out/'manifest.json')!=manifest:raise ValueError('Frozen supplement manifest required')
        if check_selector_environment()!=read(out/'reference_lock.json')['selector_environment']:
            raise ValueError('Locked selector environment differs')
        spec=next(s for s in specs if s['task_id']==args.task)
        with selection_scope(out/'tasks'/args.task/'vert_selection.jsonl',args.task):
            code=formal.run_worker(p,spec,manifest,out)
        raise SystemExit(code)
    if args.summary:
        print(json.dumps(summarize(p,specs,manifest,out),indent=2));return
    if not any((args.ready,args.batch,args.finish)):
        print(json.dumps(dict(tasks=len(specs),rounds=4500,fingerprint=manifest['fingerprint'],
            only_config_change='vert_top_k:20->0; author K-means selection',training_started=False)));return
    if args.ready or args.batch:
        lock=references(p,manifest);full.check_runtime(lock['expected_environment'])
        env=check_selector_environment()
        if args.ready:print(json.dumps(dict(ready=True,tasks=30,selector=env)));return
        from run_cifar_w85_history_admission import prepare
        prepare(out,manifest,lock)
        from cifar_vert_kmeans_scheduler import run
        run(args.config,p,specs,manifest,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if read(out/'manifest.json')!=manifest:raise ValueError('Existing supplement identity required')
    from cifar_vert_kmeans_report import finish
    print(json.dumps(finish(p,specs,manifest,out),indent=2))


if __name__=='__main__':main()

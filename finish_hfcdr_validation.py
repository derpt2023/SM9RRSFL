#!/usr/bin/env python3
"""One-shot completion pipeline: wait, audit, render and archive. Never trains."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import zipfile

import hfcdr_validation_protocol as p
from hfcdr_validation_scheduler import now


def matching_command(args,cwd):
    target=p.ROOT/'run_hfcdr_validation.py'
    return ('--batch' in args and bool(args) and Path(args[0]).name.startswith('python')
        and any((Path(arg) if Path(arg).is_absolute() else Path(cwd)/arg).resolve()==target
                for arg in args if arg and Path(arg).name=='run_hfcdr_validation.py'))


def controller_live(controller):
    cmd=Path('/proc')/str(controller['pid'])/'cmdline'
    try:
        text=cmd.read_bytes().decode(errors='replace')
        cwd=(cmd.parent/'cwd').resolve(strict=True)
    except FileNotFoundError:
        return False
    return matching_command(text.split('\0'),cwd)


def terminal_state(controller,live):
    if live:
        return 'waiting'
    if controller['status'] in ('complete','incomplete') and not controller['active']:
        return 'terminal'
    return 'unexpected_controller_exit'


def preservation(original_root,inventory):
    root=Path(original_root)
    missing=[];changed=[]
    for name,entry in inventory.items():
        path=root/name
        if not path.is_file():missing.append(name)
        elif path.stat().st_size!=entry['bytes'] or p.sha(path)!=entry['sha256']:changed.append(name)
    return dict(checked_files=len(inventory),unchanged=not missing and not changed,missing=missing,changed=changed)


def package(study):
    export=study/'export'
    export.mkdir(exist_ok=True)
    archive=export/'hfcdr_validation_attack_v1.zip'
    if archive.exists():raise FileExistsError('Existing delivery archive retained')
    skip={'finalizer.lock','finalizer.log','finalization.json','controller.lock'}
    paths=[path for path in sorted(study.rglob('*')) if path.is_file() and export not in path.parents
           and path.name not in skip and not path.name.endswith('.tmp')]
    inventory={str(path.relative_to(study)):dict(bytes=path.stat().st_size,sha256=p.sha(path)) for path in paths}
    temp=archive.with_suffix('.zip.partial')
    if temp.exists():raise FileExistsError('Existing partial archive retained for inspection')
    with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as bundle:
        for path in paths:
            relative=str(path.relative_to(study))
            bundle.write(path,arcname=relative,compress_type=zipfile.ZIP_STORED if path.suffix in ('.npz','.gz','.png') else zipfile.ZIP_DEFLATED)
        bundle.writestr('delivery_manifest.json',json.dumps(inventory,sort_keys=True,indent=2)+'\n')
    temp.rename(archive)
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=p.sha(archive),
                 files=len(inventory),file_inventory=inventory)
    p.write(export/'delivery_receipt.json',receipt)
    return {k:v for k,v in receipt.items() if k!='file_inventory'}


def finish(study,original_root,*,wait=False):
    out=p.safe_output(study)
    with (out/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'finalization.json').exists():
            raise FileExistsError('Existing finalizer evidence retained; inspect it before any new attempt')
        manifest=p.read(out/'manifest.json')
        state=dict(pid=os.getpid(),fingerprint=manifest['fingerprint'],started_at=now(),trains=False)
        p.write(out/'finalization.json',dict(state,status='waiting',at=now()))
        try:
            while True:
                c=p.read(out/'controller.json')
                if c['fingerprint']!=manifest['fingerprint']:
                    raise ValueError('Controller source identity differs')
                terminal=terminal_state(c,controller_live(c))
                if terminal=='terminal':break
                if terminal=='unexpected_controller_exit':
                    raise RuntimeError('Controller exited without an orderly terminal state; no automatic restart')
                if not wait:return dict(state,status='waiting')
                time.sleep(30)
            p.check(out)
            p.write(out/'finalization.json',dict(state,status='auditing',at=now()))
            from audit_hfcdr_validation import audit
            verified=audit(out,full=True)
            p.write(out/'provenance/final_integrity_audit.json',verified)
            before=p.read(out/'provenance/original_results_before.json')
            protected=preservation(original_root,before)
            protected['original_sources_unchanged']=all(p.sha(Path(original_root)/name)==value
                for name,value in manifest['frozen_sources'].items())
            p.write(out/'provenance/original_preservation_final.json',protected)
            if not protected['unchanged'] or not protected['original_sources_unchanged']:
                raise ValueError('Original experiment/source changed externally; retain evidence for review')
            p.write(out/'finalization.json',dict(state,status='rendering',at=now()))
            from render_hfcdr_validation import render
            destination=out/('final_evaluation' if verified['full_matrix_executed'] else 'incomplete_evaluation')
            visual=render(out,destination)
            p.write(out/'finalization.json',dict(state,status='packaging',at=now(),plots=visual['plots']))
            screening=p.read(destination/'report.json')
            receipt=package(out)
            result=dict(state,status='complete' if verified['full_matrix_executed'] else 'incomplete',at=now(),
                full_matrix_executed=verified['full_matrix_executed'],all_results_complete=verified['all_results_complete'],
                counts=verified['counts'],health_failure_count=len(verified['health_failures']),
                attack_screen_passed=screening['all_attack_settings_pass'],
                next_stage_allowed=screening['next_stage_allowed'],
                execution_or_scientific_failures=verified['execution_or_scientific_failures'],
                report=str(destination/'report.html'),delivery=receipt,automatic_retry=False)
            p.write(out/'finalization.json',result)
            return result
        except BaseException as error:
            p.write(out/'finalization.json',dict(state,status='failed',at=now(),error=repr(error),automatic_retry=False))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=p.OUTPUT)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    print(json.dumps(finish(args.study,args.original_root,wait=args.wait),indent=2))


if __name__=='__main__':main()

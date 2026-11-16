#!/usr/bin/env python3
"""One-shot completion delivery for confirmation or formal; never launches training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import zipfile

from hfcdr_pilot_protocol import ROOT,read,write,sha
from hfcdr_validation_scheduler import now
from finish_hfcdr_validation import preservation,terminal_state


def components(phase):
    if phase=='confirmation':
        import hfcdr_validation_protocol as protocol
        from hfcdr_confirmation import OUTPUT,assess_output
        from audit_hfcdr_validation import audit
        from render_hfcdr_confirmation import render
        return protocol,OUTPUT,audit,render,assess_output,'run_hfcdr_confirmation.py'
    if phase=='formal':
        import hfcdr_formal_protocol as protocol
        from audit_hfcdr_formal import audit
        from render_hfcdr_formal import render
        return protocol,protocol.OUTPUT,audit,render,None,'run_hfcdr_formal.py'
    raise ValueError('Only confirmation or formal completion supported')


def matching_command(args,cwd,program):
    target=ROOT/program
    return ('--batch' in args and bool(args) and Path(args[0]).name.startswith('python')
        and any((Path(a) if Path(a).is_absolute() else Path(cwd)/a).resolve()==target
                for a in args if a and Path(a).name==program))


def controller_live(controller,program):
    cmd=Path('/proc')/str(controller['pid'])/'cmdline'
    try:return matching_command(cmd.read_bytes().decode(errors='replace').split('\0'),(cmd.parent/'cwd').resolve(strict=True),program)
    except FileNotFoundError:return False


def package(study,phase):
    if phase not in ('confirmation','formal'):raise ValueError('Invalid archive phase')
    export=study/'export';export.mkdir(exist_ok=True)
    archive=export/('hfcdr_'+phase+'_v1.zip');temp=archive.with_suffix('.zip.partial')
    if archive.exists() or temp.exists():raise FileExistsError('Existing archive evidence retained')
    skip={'finalizer.lock','finalizer.log','finalization.json','controller.lock'}
    paths=[f for f in sorted(study.rglob('*')) if f.is_file() and export not in f.parents
           and f.name not in skip and not f.name.endswith('.tmp')]
    inventory={str(f.relative_to(study)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in paths}
    with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        for f in paths:
            z.write(f,arcname=str(f.relative_to(study)),compress_type=zipfile.ZIP_STORED
                    if f.suffix in ('.npz','.gz','.png') else zipfile.ZIP_DEFLATED)
        z.writestr('delivery_manifest.json',json.dumps(inventory,sort_keys=True,indent=2)+'\n')
    temp.rename(archive)
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        files=len(inventory),file_inventory=inventory)
    write(export/'delivery_receipt.json',receipt)
    return {k:v for k,v in receipt.items() if k!='file_inventory'}


def freeze_decision(path,decision):
    if path.exists():
        if read(path)!=decision:raise ValueError('Existing frozen confirmation differs')
    else:write(path,decision)


def finish(phase,study,original_root,*,wait=False):
    protocol,default,audit,render,assess,program=components(phase)
    out=protocol.safe_output(study or default)
    with (out/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'finalization.json').exists():raise FileExistsError('Existing finalizer evidence retained')
        manifest=read(out/'manifest.json')
        if manifest['policy']['phase']!=phase:raise ValueError('Wrong study phase')
        state=dict(pid=os.getpid(),phase=phase,fingerprint=manifest['fingerprint'],started_at=now(),trains=False)
        write(out/'finalization.json',dict(state,status='waiting',at=now()))
        try:
            while True:
                controller=read(out/'controller.json')
                if controller['fingerprint']!=manifest['fingerprint']:raise ValueError('Controller identity differs')
                terminal=terminal_state(controller,controller_live(controller,program))
                if terminal=='terminal':break
                if terminal!='waiting':raise RuntimeError('Unexpected controller exit; no restart')
                if not wait:return dict(state,status='waiting')
                time.sleep(30)
            protocol.check(out);write(out/'finalization.json',dict(state,status='auditing',at=now()))
            evidence=audit(out,full=True);write(out/'provenance/final_integrity_audit.json',evidence)
            protected=preservation(original_root,read(out/'provenance/original_results_before.json'))
            protected['original_sources_unchanged']=all(sha(Path(original_root)/n)==v for n,v in manifest['frozen_sources'].items())
            write(out/'provenance/original_preservation_final.json',protected)
            if not protected['unchanged'] or not protected['original_sources_unchanged']:raise ValueError('Protected originals changed')
            decision=None
            if assess is not None and evidence['all_results_complete']:
                decision=assess(out);freeze_decision(out/'confirmation.json',decision)
            destination=out/('final_evaluation' if evidence['all_results_complete'] else 'incomplete_evaluation')
            write(out/'finalization.json',dict(state,status='rendering',at=now()))
            visual=render(out,destination,decision) if phase=='confirmation' else render(out,destination)
            write(out/'finalization.json',dict(state,status='packaging',at=now(),plots=visual['plots']))
            receipt=package(out,phase)
            result=dict(state,status='complete' if evidence['all_results_complete'] else 'incomplete',at=now(),
                full_matrix_executed=evidence['full_matrix_executed'],all_results_complete=evidence['all_results_complete'],
                counts=evidence['counts'],health_failure_count=len(evidence['health_failures']),
                formal_allowed=bool(decision and decision['formal_allowed']) if phase=='confirmation' else None,
                execution_or_scientific_failures=evidence['execution_or_scientific_failures'],
                report=str(destination/'report.html'),delivery=receipt,automatic_retry=False,
                automatically_starts_next_phase=False)
            write(out/'finalization.json',result);return result
        except BaseException as error:
            write(out/'finalization.json',dict(state,status='failed',at=now(),error=repr(error),automatic_retry=False))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('confirmation','formal'),required=True)
    parser.add_argument('--study',type=Path)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    print(json.dumps(finish(args.phase,args.study,args.original_root,wait=args.wait),indent=2))

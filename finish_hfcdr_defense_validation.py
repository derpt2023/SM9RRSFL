#!/usr/bin/env python3
"""Wait, audit, freeze bounded choices, render and archive; never trains."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import zipfile

import hfcdr_validation_protocol as p
import hfcdr_validation_selection as selection
from hfcdr_validation_scheduler import now
from finish_hfcdr_validation import preservation, terminal_state


def matching_command(args, cwd):
    target = p.ROOT/'run_hfcdr_defense_selection.py'
    return ('--batch' in args and bool(args) and Path(args[0]).name.startswith('python')
        and any((Path(a) if Path(a).is_absolute() else Path(cwd)/a).resolve() == target
                for a in args if a and Path(a).name == target.name))


def controller_live(controller):
    cmd = Path('/proc')/str(controller['pid'])/'cmdline'
    try: return matching_command(cmd.read_bytes().decode(errors='replace').split('\0'), (cmd.parent/'cwd').resolve(strict=True))
    except FileNotFoundError: return False


def package(study):
    export = study/'export'; export.mkdir(exist_ok=True)
    archive = export/'hfcdr_validation_defense_v1.zip'; temp = archive.with_suffix('.zip.partial')
    if archive.exists() or temp.exists(): raise FileExistsError('Existing archive evidence retained')
    skip = {'finalizer.lock', 'finalizer.log', 'finalization.json', 'controller.lock'}
    paths = [f for f in sorted(study.rglob('*')) if f.is_file() and export not in f.parents
             and f.name not in skip and not f.name.endswith('.tmp')]
    inventory = {str(f.relative_to(study)):dict(bytes=f.stat().st_size, sha256=p.sha(f)) for f in paths}
    with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as z:
        for f in paths:
            z.write(f, arcname=str(f.relative_to(study)), compress_type=zipfile.ZIP_STORED
                    if f.suffix in ('.npz', '.gz', '.png') else zipfile.ZIP_DEFLATED)
        z.writestr('delivery_manifest.json', json.dumps(inventory, sort_keys=True, indent=2)+'\n')
    temp.rename(archive)
    receipt = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=p.sha(archive),
                   files=len(inventory), file_inventory=inventory)
    p.write(export/'delivery_receipt.json', receipt)
    return {k:v for k,v in receipt.items() if k != 'file_inventory'}


def finish(study, attack, original_root, *, wait=False):
    out = p.safe_output(study)
    with (out/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'finalization.json').exists(): raise FileExistsError('Existing finalizer evidence retained')
        manifest = p.read(out/'manifest.json')
        if manifest['policy']['phase'] != 'defense_search': raise ValueError('Defense stage required')
        state = dict(pid=os.getpid(), fingerprint=manifest['fingerprint'], started_at=now(), trains=False)
        p.write(out/'finalization.json', dict(state, status='waiting', at=now()))
        try:
            while True:
                c = p.read(out/'controller.json')
                if c['fingerprint'] != manifest['fingerprint']: raise ValueError('Controller identity differs')
                terminal = terminal_state(c, controller_live(c))
                if terminal == 'terminal': break
                if terminal != 'waiting': raise RuntimeError('Unexpected controller exit; no restart')
                if not wait: return dict(state, status='waiting')
                time.sleep(30)
            p.check(out); p.write(out/'finalization.json', dict(state, status='auditing', at=now()))
            from audit_hfcdr_validation import audit
            evidence = audit(out, full=True)
            p.write(out/'provenance/final_integrity_audit.json', evidence)
            protected = preservation(original_root, p.read(out/'provenance/original_results_before.json'))
            protected['original_sources_unchanged'] = all(p.sha(Path(original_root)/n) == v for n,v in manifest['frozen_sources'].items())
            p.write(out/'provenance/original_preservation_final.json', protected)
            if not protected['unchanged'] or not protected['original_sources_unchanged']:
                raise ValueError('Protected original source/result changed')
            choice = None
            if evidence['all_results_complete']:
                choice = selection.select(attack, out)
                if (out/'selection.json').exists(): raise FileExistsError('Existing selected configuration retained')
                p.write(out/'selection.json', choice)
            p.write(out/'finalization.json', dict(state, status='rendering', at=now()))
            from render_hfcdr_defense_validation import render
            destination = out/('final_evaluation' if evidence['all_results_complete'] else 'incomplete_evaluation')
            visual = render(out, destination, choice)
            p.write(out/'finalization.json', dict(state, status='packaging', at=now(), plots=visual['plots']))
            receipt = package(out)
            result = dict(state, status='complete' if evidence['all_results_complete'] else 'incomplete', at=now(),
                full_matrix_executed=evidence['full_matrix_executed'], all_results_complete=evidence['all_results_complete'],
                counts=evidence['counts'], health_failure_count=len(evidence['health_failures']),
                confirmation_allowed=bool(choice and choice['confirmation_allowed']),
                execution_or_scientific_failures=evidence['execution_or_scientific_failures'],
                report=str(destination/'report.html'), delivery=receipt, automatic_retry=False,
                automatically_starts_confirmation=False)
            p.write(out/'finalization.json', result); return result
        except BaseException as error:
            p.write(out/'finalization.json', dict(state, status='failed', at=now(), error=repr(error), automatic_retry=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=selection.OUTPUT)
    parser.add_argument('--attack-output', type=Path, default=p.OUTPUT)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    print(json.dumps(finish(args.study, args.attack_output, args.original_root, wait=args.wait), indent=2))

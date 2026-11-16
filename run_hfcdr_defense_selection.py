#!/usr/bin/env python3
"""Gated next-phase entry. Never starts training when FedAvg screening fails."""
import argparse
import json
from pathlib import Path
import hfcdr_validation_protocol as p
import hfcdr_validation_selection as s


def prepare(attack,output,data_root,spec_path=s.SPEC):
    spec=p.specification(spec_path);s.verify_selection_sources(spec)
    gate=s.attack_gate(attack,spec)
    result=p.prepare(output,data_root,spec_path)
    p.write(p.safe_output(output)/'entry_gate.json',gate)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--batch',action='store_true')
    mode.add_argument('--select',action='store_true')
    parser.add_argument('--attack-output',type=Path,default=p.OUTPUT)
    parser.add_argument('--output',type=Path,default=s.OUTPUT)
    parser.add_argument('--data-root',type=Path,default=p.ROOT/'data')
    args=parser.parse_args(argv)
    spec=p.specification(s.SPEC);s.verify_selection_sources(spec)
    if args.prepare:
        result=prepare(args.attack_output,args.output,args.data_root)
    elif args.batch:
        out=p.safe_output(args.output);p.check(out)
        if p.read(out/'entry_gate.json')!=s.attack_gate(args.attack_output,spec):
            raise ValueError('Upstream attack gate changed')
        from hfcdr_validation_scheduler import batch
        batch(out,args.data_root);return 0
    elif args.select:
        result=s.select(args.attack_output,args.output)
        path=p.safe_output(args.output)/'selection.json'
        if path.exists():raise FileExistsError('Selection is already frozen; preserve its evidence')
        p.write(path,result)
    else:
        plan=p.plan(s.SPEC)
        result=dict(plan['audit'],training_started=False,requires_passed_attack_gate=True,
            frozen_selection_sources=True,fingerprint=plan['fingerprint'])
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())

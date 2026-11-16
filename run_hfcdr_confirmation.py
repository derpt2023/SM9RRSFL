#!/usr/bin/env python3
"""Gated independent validation; default plans only, never opens official test."""
import argparse
import json
from pathlib import Path
import hfcdr_validation_protocol as p
import hfcdr_validation_selection as s
import hfcdr_confirmation as c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--build-config',action='store_true')
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--batch',action='store_true')
    mode.add_argument('--assess',action='store_true')
    parser.add_argument('--attack-output',type=Path,default=p.OUTPUT)
    parser.add_argument('--defense-output',type=Path,default=s.OUTPUT)
    parser.add_argument('--output',type=Path,default=c.OUTPUT)
    parser.add_argument('--data-root',type=Path,default=p.ROOT/'data')
    args = parser.parse_args(argv)
    if args.build_config:
        result = c.build(args.attack_output,args.defense_output)
    elif args.prepare:
        result = c.prepare(args.attack_output,args.defense_output,args.output,args.data_root)
    elif args.batch:
        expected = c.upstream(args.attack_output,args.defense_output)
        p.check(args.output)
        if p.read(args.output/'manifest.json')['policy'] != expected or p.read(args.output/'entry_gate.json') != expected['entry_gate']:
            raise ValueError('Confirmation upstream gate changed')
        from hfcdr_validation_scheduler import batch
        batch(args.output,args.data_root); return 0
    elif args.assess:
        result = c.assess_output(args.output)
        path = args.output/'confirmation.json'
        if path.exists(): raise FileExistsError('Existing confirmation evidence retained')
        p.write(path,result)
    else:
        r = c.rules()
        result = dict(expected_tasks=312,seed=r['seed'],methods=list(c.METHODS),training_started=False,
            official_test_used=False,requires_complete_frozen_defense_selection=True,
            configuration_materialized=c.SPEC.exists())
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())

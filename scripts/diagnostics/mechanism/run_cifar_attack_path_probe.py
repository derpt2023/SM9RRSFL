#!/usr/bin/env python3
"""Two fresh serial H0 Dirichlet alternating-attack backward probes."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import cifar_attack_path_protocol as protocol
import cifar_attack_path_runtime as runtime
import cifar_attack_path_observer as observer
def worker(args):
 m,tasks=protocol.read_plan(args.output);task=next(t for t in tasks if t["task_id"]==args.worker);folder=args.output/"tasks"/task["task_id"];attempt=folder/"attempts"/time.strftime("%Y%m%dT%H%M%S");attempt.mkdir(parents=True,exist_ok=False)
 try:
  from run_cifar_six_from_scratch import load_split
  from sm9rrsfl import fl
  split,contract=load_split({"dataset":{"name":"cifar10","data_dir":str(args.data_dir or "data/cifar10"),"download":True,"split_seed":20260917}} ,args.data_dir)
  if m.get("data_contract") is not None and contract!=m["data_contract"]:raise ValueError("CIFAR data contract changed")
  cfg=fl.ExperimentConfig(**task["config"])
  with observer.observe(task) as obs:result=fl.run_experiment(split.calibration_dataset,cfg)
  payload={"status":"complete","task_fingerprint":task["fingerprint"],"gpu_uuid":task.get("same_gpu_uuid"),"observations":obs.finish(result),"fresh_start":True,"checkpoints_used":False,"completed_training_rounds":result.stopped_round,"requested_training_rounds":cfg.rounds,"wall_seconds":0.0}
  runtime.write_json(attempt/"completed.json",payload);runtime.write_json(folder/"completed.json",payload);print("COMPLETED_ATTACK_PATH "+task["task_id"]);return 0
 except BaseException as e:
  runtime.write_json(attempt/"failure.json",{"status":"failed_attack_path","task_id":task["task_id"],"exception":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()});raise
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=protocol.DEFAULT_OUTPUT);p.add_argument("--clean-output",type=Path,default=protocol.DEFAULT_CLEAN);p.add_argument("--data-dir",type=Path);p.add_argument("--summary",action="store_true");p.add_argument("--plan-only",action="store_true");p.add_argument("--worker",help=argparse.SUPPRESS);a=p.parse_args(argv);a.output=a.output.resolve()
 if a.summary:
  from cifar_attack_path_report import summarize,print_summary;print_summary(summarize(a.output));return 0
 if a.plan_only:
  print(json.dumps({"protocol":protocol.PROTOCOL,"fresh_processes":2,"rounds_per_process":27,"attack_start_round":25,"partition":"dirichlet","malicious_ratio":.1,"stage_scope":observer.STAGES,"training_started":False,"automatic_next_stage":False},indent=2));return 0
 if a.worker:return worker(a)
 if not (a.output/"manifest.json").exists():
  ref={"same_gpu_uuid":os.environ.get("CUDA_VISIBLE_DEVICES"),"data_contract":None};m=protocol.build_manifest(ref);protocol.write_plan(a.output,m,protocol.build_tasks(m))
 m,tasks=protocol.read_plan(a.output)
 for task in tasks:
  if runtime.load_completed(a.output,task) is not None:continue
  cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker",task["task_id"],"--output",str(a.output)];
  if a.data_dir:cmd += ["--data-dir",str(a.data_dir)]
  if subprocess.call(cmd,cwd=Path(__file__).resolve().parent,env=dict(os.environ,CUDA_VISIBLE_DEVICES=m.get("same_gpu_uuid") or os.environ.get("CUDA_VISIBLE_DEVICES","0"))):return 1
 from cifar_attack_path_report import summarize,print_summary;v=summarize(a.output);print_summary(v);return 0 if v["status"]=="complete" else 2
if __name__=="__main__":raise SystemExit(main())

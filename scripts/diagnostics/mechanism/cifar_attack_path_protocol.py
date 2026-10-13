"""Independent short alternating-attack backward-path probe."""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json
from pathlib import Path
from sm9rrsfl import fl
REPO=Path(__file__).resolve().parent; PROTOCOL="cifar-attack-path-observer-v1"; DEFAULT_OUTPUT=REPO/"outputs/cifar_v8_diagnostic_v1/attack_path_v1"; DEFAULT_CLEAN=REPO/"outputs/cifar_v8_diagnostic_v1/clean_horizon80_v1"; ROUNDS=27; ATTACK_START=25
NEW_SOURCES=("cifar_attack_path_protocol.py","cifar_attack_path_observer.py","cifar_attack_path_runtime.py","cifar_attack_path_report.py","run_cifar_attack_path_probe.py")
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def config_dict():return asdict(fl.ExperimentConfig(method="sm9rrs",malicious_ratio=.1,num_clients=100,rounds=ROUNDS,partition="dirichlet",dirichlet_alpha=.5,seed=2026093001,compute_backend="torch",device="cuda:0",crypto_mode="simulated",batch_size=50,local_epochs=1,lr=.05,lr_decay=.99,detector_window=20,attack="alternating_minimization",attack_start_round=ATTACK_START,attack_boost=5.,attack_epochs=1,attack_stealth_steps=1,attack_distance_weight=.0001,attack_source_label=5,attack_target_label=7,attack_target_count=200,checkpoint_interval=1,eval_interval=1,early_stop=False,sm9_workers=1))
def source_hashes():
    names=NEW_SOURCES+("sm9rrsfl/fl.py","sm9rrsfl/torch_backend.py");return {n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in names}
def build_manifest(reference):
    v={"protocol":PROTOCOL,"reference":reference,"rounds":ROUNDS,"attack_start_round":ATTACK_START,"config":config_dict(),"same_gpu_uuid":reference.get("same_gpu_uuid"),"data_contract":reference.get("data_contract"),"execution_order":["attack_path_dirichlet_repeat1","attack_path_dirichlet_repeat2"],"fresh_process_per_task":True,"serial_same_physical_gpu":True,"checkpoints_reused":False,"formal_qualification_assessed":False,"automatic_next_stage":False,"scope":["benign_reference","stealth_benign","target_adversarial"]};v["source_sha256"]=source_hashes();v["fingerprint"]=digest(v);return v
def build_tasks(m):
    out=[]
    for r in (1,2):
        for policy in ("original","attacker_backward_cudnn_deterministic"):
            tid=f"attack_path_{policy}_repeat{r}"; body={"task_id":tid,"repeat":r,"policy":policy,"config":m["config"],"same_gpu_uuid":m.get("same_gpu_uuid"),"purpose":"short_alternating_attack_backward_path_probe"}; out.append({**body,"fingerprint":digest(body)})
    return out
def write_plan(out,m,tasks):
    out.mkdir(parents=True,exist_ok=True);(out/"task_plans").mkdir(exist_ok=True);(out/"manifest.json").write_text(json.dumps(m,sort_keys=True,indent=2)+"\n");(out/"task_plans/attack_path.json").write_text(json.dumps({"manifest_fingerprint":m["fingerprint"],"tasks":tasks},sort_keys=True,indent=2)+"\n")
    for t in tasks:(out/"tasks"/t["task_id"]).mkdir(parents=True,exist_ok=True);(out/"tasks"/t["task_id"] / "task.json").write_text(json.dumps(t,sort_keys=True,indent=2)+"\n")
def read_plan(out):
    out=Path(out);m=json.loads((out/"manifest.json").read_text());
    if m.get("protocol")!=PROTOCOL or digest({k:v for k,v in m.items() if k!="fingerprint"})!=m.get("fingerprint"):raise ValueError("attack-path manifest identity mismatch")
    if source_hashes()!=m.get("source_sha256"):raise ValueError("attack-path source identity changed")
    t=build_tasks(m);p=json.loads((out/"task_plans/attack_path.json").read_text());
    if p!={"manifest_fingerprint":m["fingerprint"],"tasks":t}:raise ValueError("attack-path task plan mismatch")
    return m,t

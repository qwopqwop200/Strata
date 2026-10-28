from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parent;R=D.parent;U=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
origin=U/'criticalpath-IQ1M-decode-v1/validation-iq1m-swizzle/run_gpu.py'
a=origin.read_text().replace('DS4_IQ1M_TABLE_SWIZZLE','DS4_PREFILL_ROUTER_REUSE').replace("text.count('dispatch_selected=1')==2","text.count('dispatch_selected=1')==1").replace('PASS actual_IQ1M_NE4 bits=6 timings=8','PASS actual_prefill_router bits=6 timings=8')
assert a==(D/'run_gpu.py').read_text()
(D/'RUNNER-PROVENANCE.json').write_text(json.dumps(dict(source=str(origin),source_sha256=sha(origin),exact_transform='only env name, one dispatch marker instead of2, final PASS text',existing_lifecycle_tests_reused='IQ runner CPU5 and actual source/payload/reap PASS; this new component has not executed',no_new_lifecycle_guarantees=True),indent=2)+'\n')
paths=[p for root in [R/'src/dsv41',R/'include',R/'third_party/ggml'] for p in root.rglob('*')if p.is_file()and'.git'not in p.parts]
paths += [p for p in D.iterdir() if p.is_file() and p.name not in ['FREEZE.json','READY.json']]
paths += [Path('/home/qwopqwop200/models/Abiray-DeepSeek-V4.1-IQ2_XXS/maya-ds4-q4/tensors.txt')]
f=dict(writer_released=True,sha256={str(p):sha(p)for p in sorted(set(paths))});(D/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n')
r=dict(status='READY_COMPONENT_ONLY',writer_released=True,source_sha256=sha(R/'src/dsv41/ds4_kernels.cu'),freeze_sha256=sha(D/'FREEZE.json'),cpu_source_tests=6,gpu_compile_run=0,common_lock=str(U.parent/'.validation.lock'),STOP=str(U/'STOP'),command=['python3','-B',str(D/'run_gpu.py')],budget_s=120,compile_jobs=1,GPU_runs=1,bit_cases=6,timing_batches=8,max_controlled_host_gpu_bytes=1<<30,model_jobs=0,activation={'env':'DS4_PREFILL_ROUTER_REUSE','value':'1','prefix':'ds4 prefill router reuse:','required':'dispatch_selected=1 tokens_per_warp=2 first_T=17 n=384 k=5120 ldy=1024 scope=process_first_dispatch kernel_completion_certified=0','scope_limit':'first dispatch; not completion/frequency/gain'})
(D/'READY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

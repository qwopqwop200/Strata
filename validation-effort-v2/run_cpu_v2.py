from pathlib import Path
import os,json,hashlib,fcntl,subprocess,signal,ctypes,time,resource
D=Path(__file__).resolve().parent;R=D.parent;U=R.parent;PYTHON='/home/qwopqwop200/Strata-spark/.venv/bin/python';O=D/'cpu-v2';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[R/'serve/server.py',R/'serve/test_atomic_effort.py',R/'tools/strata_tokenizer.py',R/'serve/frontend.py',R/'serve/dsml.py',*(R/'atomic_regression').glob('*.py'),U/'atomic-agent1-pack-v1/header-probes/TOKENIZER-KV.json',U/'atomic-agent1-pack-v1/header-probes/TOKENIZER-SOURCE.json',Path('/home/qwopqwop200/models/AtomicChat-DeepSeek-V4.1-IQ2_XXS/original/chat_template.jinja'),U/'llama-pinned/src/llama-vocab.cpp']
with (U.parent/'.validation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (U/'STOP').exists():raise RuntimeError('STOP')
 before={str(p):sha(p) for p in paths};start=time.monotonic();parent=os.getpid()
 def setup():
  if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
  resource.setrlimit(resource.RLIMIT_AS,(768*2**20,768*2**20))
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');cmd=[PYTHON,'-B','-m','unittest','serve.test_atomic_effort','-v']
 with (O/'run.log').open('w') as log:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True,preexec_fn=setup)
  timed_out=False
  try:rc=p.wait(timeout=55)
  except subprocess.TimeoutExpired:timed_out=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=3)
 after={str(p):sha(p) for p in paths};result={'status':'PASS' if rc==0 and before==after and not timed_out else 'FAIL','command':cmd,'owned_pid':p.pid,'owned_returncode':rc,'owned_reaped':p.returncode is not None,'timeout':timed_out,'source_prepost_equal':before==after,'source_sha256':before,'elapsed_lock_s':time.monotonic()-start,'maxrss_child_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'CUDA_runs':0,'model_runs':0,'scope':'Actual API HTTP mocks/Service renderer using Atomic header/original template; no real model'}
 (O/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if result['status']=='PASS' else 2)

"""Parent-authorized bounded CPU validation; SAME lock/PDEATHSIG/owned process group."""
import ctypes,fcntl,hashlib,importlib.util,json,os,pathlib,signal,subprocess,time
R=pathlib.Path(__file__).resolve().parents[1];P=R/'validation-context';U=R.parent
HELPER=U/'criticalpath-q5-preload-v1/validation-q5-preload/gpu-component-v3/run_cuda.py'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 out=P/'cpu-attempt1';out.mkdir(exist_ok=False)
 inputs=[R/'src/dsv41/ds4_context_capacity.hpp',R/'src/dsv41/ds4_model.cu',R/'src/dsv41/ds4_main.cpp',*P.glob('*.py'),P/'test_capacity.cpp',* (R/'tools/atomic_context').glob('*.py'),HELPER];before={str(p):sha(p)for p in inputs}
 spec=importlib.util.spec_from_file_location('context_owned_helper',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 with (U.parent/'.validation.lock').open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not(U/'STOP').exists();start=time.monotonic();deadline=start+30;owner=os.getpid();result=dict(status='FAIL',CPU_compiles=0,CPU_runs=0,CUDA_GPU_model_runs=0,capacity_PASS=False)
  def job(cmd,name):
   assert not(U/'STOP').exists();assert all(sha(p)==h for p,h in before.items());(out/(name+'.argv.json')).write_text(json.dumps(cmd,indent=2)+'\n')
   with (out/(name+'.log')).open('x')as log:
    child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),),preexec_fn=helper.child_setup(owner),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    assert helper.wait_owned(child,max(.01,deadline-time.monotonic()),out/(name+'-POST-PID.json'))==0,name
  try:
   result['CPU_compiles']=1;job(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Isrc/dsv41',str(P/'test_capacity.cpp'),'-o',str(out/'component')],'compile')
   result['CPU_runs']=1;job([str(out/'component')],'component')
   job(['python3',str(P/'test_contract.py')],'python-tests');job(['python3',str(P/'source_proof.py')],'inverse')
   text=(out/'component.log').read_text();import sys;sys.path.insert(0,str(R/'tools/atomic_context'));from capacity import fingerprint
   assert 'FNV64 '+fingerprint([0,1,128803,129279])in text
   result['status']='PASS'
  except BaseException as e:result['error']=repr(e)
  finally:
   result['source_unchanged']=all(sha(p)==h for p,h in before.items());result['input_sha256']=before;result['elapsed_after_lock_s']=time.monotonic()-start
   if not result['source_unchanged']:result['status']='FAIL'
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
  raise SystemExit(0 if result['status']=='PASS'else 1)

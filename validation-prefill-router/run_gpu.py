import pathlib,json,hashlib,fcntl,subprocess,os,signal,time,ctypes,shlex,resource
D=pathlib.Path(__file__).resolve().parent;R=D.parent;U=R.parent;STOP=U/'STOP';LOCK=U.parent/'.validation.lock'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def guard():
 for p,h in json.loads((D/'FREEZE.json').read_text())['sha256'].items():
  if sha(p)!=h:raise RuntimeError('frozen input changed '+p)
def payload():
 rows=[]
 for c in json.loads((D/'PAYLOAD.json').read_text()):
  p=pathlib.Path(c['path']);st=p.stat();h=hashlib.sha256()
  with p.open('rb') as f:
   f.seek(c['offset']);left=c['bytes']
   while left:
    b=f.read(min(left,1024*1024))
    if not b:raise RuntimeError('short payload')
    h.update(b);left-=len(b)
  rows.append(dict(c,sha256=h.hexdigest(),dev=st.st_dev,inode=st.st_ino,file_size=st.st_size,mtime_ns=st.st_mtime_ns))
 return rows
def dependencies(path):
 text=path.read_text().replace('\\\n',' ').partition(':')[2];r=[]
 for name in shlex.split(text):
  p=pathlib.Path(name);r.append((p if p.is_absolute() else R/p).resolve())
 return sorted(set(r))
def group_members(pgid):
 live=[];zombies=[]
 for task in pathlib.Path('/proc').iterdir():
  if not task.name.isdigit():continue
  try:
   fields=(task/'stat').read_text().rpartition(')')[2].split()
   if int(fields[2])==pgid:(zombies if fields[0]=='Z' else live).append(int(task.name))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return live,zombies

def wait_owned(p,limit,record):
 timed_out=False;reaped=False;code=None;exception=None
 try:code=p.wait(timeout=limit);reaped=True
 except BaseException as e:
  exception=e;timed_out=isinstance(e,subprocess.TimeoutExpired)
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  try:code=p.wait(timeout=2);reaped=True
  except subprocess.TimeoutExpired:pass
 initial,zombies=group_members(p.pid)
 if initial:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  until=time.monotonic()+1
  while time.monotonic()<until:
   live,zombies=group_members(p.pid)
   if not live:break
   time.sleep(.01)
 live,zombies=group_members(p.pid)
 record.write_text(json.dumps(dict(owned_pid=p.pid,owned_returncode=code,owned_reaped=reaped,timed_out=timed_out,residual_live_initial=initial,residual_live_after=live,residual_zombies_not_owned=zombies),indent=2)+'\n')
 if exception:raise exception
 if not reaped or initial or live:raise RuntimeError('child lifetime failed; direct reap or process-group cleanup incomplete')
 return code

def final_audit(result):
 ok=result.get('source_prepost_equal_on_exit') is True and result.get('payload_prepost_equal_on_exit') is True
 if not ok:
  result['status']='FAIL';result['final_audit_error']='source/payload post-audit failed'
  result['source_guard']=result.get('source_prepost_equal_on_exit') is True
  result['payload_guard']=result.get('payload_prepost_equal_on_exit') is True
 return ok

def main():
 O=D/'attempt1';O.mkdir(exist_ok=False);result=dict(status='waiting_lock',models=0,compile_jobs=0,GPU_runs=0);(O/'RESULT.json').write_text(json.dumps(result))
 with LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);start=time.monotonic()
  def job(label,cmd,limit,env=None):
   if STOP.exists():raise RuntimeError('STOP after lock/between jobs')
   guard();remaining=117-(time.monotonic()-start)
   if remaining<=0:raise RuntimeError('total budget')
   (O/(label+'.argv.json')).write_text(json.dumps(cmd,indent=2)+'\n');parent=os.getpid()
   def setup():
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
   with (O/(label+'.log')).open('w') as log:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True,preexec_fn=setup)
    code=wait_owned(p,min(limit,remaining),O/(label+'-POST-PID.json'))
   guard()
   if code:raise RuntimeError(label+' exit '+str(code))
  try:
   if STOP.exists():raise RuntimeError('STOP after lock')
   guard();before=payload();(O/'PAYLOAD-BEFORE.json').write_text(json.dumps(before,indent=2)+'\n')
   includes=['-I'+str(R/'src/dsv41'),'-I'+str(R/'include'),'-I'+str(R/'third_party/ggml'),'-I'+str(D)]
   flags=['/usr/local/cuda/bin/nvcc','-std=c++20','-O3','-arch=sm_121']
   result['dependency_preprocessors']=1;job('dependencies',[*flags,'-M','-MF',str(O/'precompile.d'),*includes,str(D/'component.cu')],20)
   actual_deps=dependencies(O/'precompile.d')
   if not actual_deps:raise RuntimeError('empty dependency discovery')
   frozen=json.loads((D/'FREEZE.json').read_text())['sha256']
   for p in actual_deps:
    if p.is_relative_to(R) and str(p) not in frozen:raise RuntimeError('project dependency absent from FREEZE '+str(p))
   dependency_before={str(p):sha(p) for p in actual_deps};(O/'NVCC-DEPS-BEFORE.json').write_text(json.dumps(dependency_before,indent=2)+'\n')
   result['compile_jobs']=1;job('compile',[*flags,'-Xptxas=-v','-MD','-MF',str(O/'compile.d'),*includes,str(D/'component.cu'),'-o',str(O/'component')],60)
   actual_after=dependencies(O/'compile.d');deps={str(p):sha(p) for p in actual_after}
   if actual_deps!=actual_after or deps!=dependency_before:raise RuntimeError('actual compile dependencies mismatch/drift')
   (O/'NVCC-DEPS.json').write_text(json.dumps(dict(all_prepost_equal=True,sha256=deps),indent=2)+'\n')
   clean={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
   # Eligibility-only children return before any CUDA API/context or allocation.
   for label,value in [('unset',None),('zero','0'),('other','2')]:
    env=dict(clean)
    if value is not None:env['DS4_PREFILL_ROUTER_REUSE']=value
    job('OFF-'+label,[str(O/'component'),'--off-guard'],3,env)
   env=dict(clean,DS4_PREFILL_ROUTER_REUSE='1');result['GPU_runs']=1;job('run',[str(O/'component')],30,env)
   text=(O/'run.log').read_text();assert text.count('BITS ')==6 and text.count('TIME ')==8 and text.count('dispatch_selected=1')==1 and 'PASS actual_prefill_router bits=6 timings=8' in text
   after=payload();assert before==after;guard();(O/'PAYLOAD-AFTER.json').write_text(json.dumps(after,indent=2)+'\n');result.update(status='PASS',source_guard=True,payload_guard=True,bit_checks=6,timing_batches=8,component_sha256=sha(O/'component'))
  except BaseException as e:result.update(status='FAIL',error=repr(e));raise
  finally:
   already_failed=result.get('status')=='FAIL'
   try:guard();result['source_prepost_equal_on_exit']=True
   except BaseException as source_error:result['source_prepost_equal_on_exit']=False;result['source_after_error']=repr(source_error)
   try:
    after=payload();(O/'PAYLOAD-AFTER-ON-EXIT.json').write_text(json.dumps(after,indent=2)+'\n');result['payload_prepost_equal_on_exit']=before==after
   except BaseException as payload_error:result['payload_after_error']=repr(payload_error)
   audit_ok=final_audit(result)
   result['maxrss_children_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;result['elapsed_lock_s']=time.monotonic()-start;(O/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
   if not audit_ok and not already_failed:raise RuntimeError('final source/payload audit failed')
if __name__=='__main__':main()

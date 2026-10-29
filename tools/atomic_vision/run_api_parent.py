#!/usr/bin/env python3
"""Parent queued localhost API/browser image window; one engine, Dlock held through all reaping."""
import argparse,ctypes,hashlib,importlib.util,json,os,queue,signal,subprocess,sys,threading,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];U=R.parent;CORE=U/'atomic-agent2-vision-v1/tools/atomic_vision';sys.path.insert(0,str(CORE))
from run_geni import child_setup,guards,payload_guard
from resources import snapshot,critical
from owned import group_members
LOCK=U.parent/'.validation.lock';STOP=U/'STOP';PROFILE=Path('/home/qwopqwop200/.local/state/strata/deepseek41-q4/expert-profile.bin')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--geni-config',required=True,type=Path);ap.add_argument('--api-config',required=True,type=Path);ap.add_argument('--fixtures',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--hold-seconds',type=int,default=300);ap.add_argument('--parent-approved-model',action='store_true');ap.add_argument('--release',required=True,type=Path);a=ap.parse_args()
 if not a.parent_approved_model or not 0<=a.hold_seconds<=600:raise ValueError('bounded parent queue only')
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);gcfg=json.loads(a.geni_config.read_text());cfg=json.loads(a.api_config.read_text());release=json.loads(a.release.read_text());result={'AtomicPASS':False,'label':'OLD_MODEL_VISION_PREPARATION','pass_':False,'browser_quality_certified_by_this_driver':False,'core_config_sha256':sha(a.geni_config),'API_config_sha256':sha(a.api_config),'python_source_root':str(R),'python_source_commit':subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),'CPP_binary_sha256':gcfg['binary_sha256']};server=None;engine=None;pump=None;log=None;returncode=None;state={'EOF':False,'error':None};start=time.monotonic();deadline=start+1800
 def save(): (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 import fcntl
 with LOCK.open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if STOP.exists():raise RuntimeError('STOP after lock')
  if ctypes.CDLL(None).prctl(36,1,0,0,0)!=0:raise RuntimeError('owned child subreaper unavailable')
  g={**gcfg['guards'],**release['sha256']};guards(g);payload_guard(gcfg['payload_stat_guard']);before=snapshot(os.getpid());result['resource_before']=before
  profile=a.out/'private-state/expert-profile.bin';profile.parent.mkdir()
  if sha(PROFILE)!='8251175a83924ff291b58f14c893b36f5a7989c79fb78f04ab86063e42d7c51f':raise ValueError('original825 seed changed')
  profile.write_bytes(PROFILE.read_bytes());cfg['args'][cfg['args'].index('--expert-profile')+1]=str(profile);cfg['cwd']=str(R);cfg['log']=str(a.out/'engine.log');effective=a.out/'effective-api-config.json';effective.write_text(json.dumps(cfg,indent=2)+'\n');ready=a.out/'server-ready.json';audit=a.out/'engine-audit.jsonl'
  env={k:v for k,v in os.environ.items()if not k.startswith(('DS4_','STRATA_DS4_'))};env.update(STRATA_VALIDATION_LOCK_FD=str(lock.fileno()),STRATA_VALIDATION_AUDIT=str(audit),STRATA_DROP_CACHE_ON_EXIT='0',STRATA_VALIDATION_TOKENIZER=cfg['tokenizer'])
  try:
   parent=os.getpid();cmd=[sys.executable,'-B',str(R/'tools/atomic_vision/server_entry.py'),'--engine','strata','--config',str(effective),'--host','127.0.0.1','--port','0','--ready-file',str(ready),'--shutdown-timeout','120','--offline'];log=(a.out/'server.log').open('w');server=subprocess.Popen(cmd,cwd=str(R),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,pass_fds=(lock.fileno(),),preexec_fn=lambda:child_setup(parent));result['server_pid']=server.pid
   def reader():
    try:
     for line in server.stdout:log.write(line);log.flush()
     state['EOF']=True
    except Exception as e:state['error']=repr(e)
   pump=threading.Thread(target=reader,daemon=True);pump.start();until=time.monotonic()+180
   while not ready.exists():
    if server.poll()is not None:raise RuntimeError('API server exited before ready')
    if time.monotonic()>until:raise TimeoutError('API startup180s')
    time.sleep(.1)
   live=json.loads(ready.read_text());events=[json.loads(x)for x in audit.read_text().splitlines()];engines=[x for x in events if x['event']=='engine_ready']
   if len(engines)!=1:raise RuntimeError('one actual model engine required')
   engine=engines[0]['pid'];engine_snap=snapshot(engine)
   if engine_snap['exe']!=gcfg['binary']or sha(engine_snap['exe'])!=gcfg['binary_sha256']:raise ValueError('actual API engine differs')
   if engines[0]['info'].get('vision')!='ds4'or engines[0]['info'].get('vision_required')!=1:raise RuntimeError('native strict vision unavailable')
   fdinfo=(Path('/proc')/str(engine)/'fdinfo'/str(lock.fileno())).read_text()
   if 'FLOCK'in fdinfo and 'WRITE'in fdinfo:result['engine_inherited_exclusive_lock']=True
   else:raise RuntimeError('engine lacks inherited exclusive Dlock')
   endpoint=live['api_url']+'/v1/chat/completions';result['engine_pid']=engine;result['engine_ready']=engine_snap;result['endpoint']=endpoint
   (a.out/'API-LIVE.json').write_text(json.dumps(dict(endpoint=endpoint,browser_url=live['api_url'],engine_pid=engine,server_pid=server.pid,source=str(R),source_commit=result['python_source_commit'],label='OLD_MODEL_VISION_PREPARATION',AtomicPASS=False,UI_DONE=str(a.out/'UI_DONE')),indent=2)+'\n');save()
   client=load('new_frozen_vision_API_client',R/'tools/atomic_vision/run_api.py');oldargv=sys.argv;sys.argv=[str(R/'tools/atomic_vision/run_api.py'),'--endpoint',endpoint,'--model',cfg['model_name'],'--manifest',str(a.fixtures),'--engine-pid',str(engine),'--lock-owner-pid',str(os.getpid()),'--binary-sha256',gcfg['binary_sha256'],'--out',str(a.out/'automated-api'),'--label','OLD_MODEL_VISION_PREPARATION']
   try:client_rc=client.main()
   finally:sys.argv=oldargv
   result['automated_API_returncode']=client_rc;(a.out/'API-AUTOMATED-DONE.json').write_text(json.dumps({'client_rc':client_rc,'UI_DONE':str(a.out/'UI_DONE')})+'\n');save()
   if client_rc!=0:raise RuntimeError('actual image API content/reuse/cancel followup failed')
   until=min(deadline,time.monotonic()+a.hold_seconds)
   while time.monotonic()<until and not(a.out/'UI_DONE').exists():
    if STOP.exists():break
    if server.poll()is not None:raise RuntimeError('API server disappeared during UI window')
    time.sleep(.2)
   result['UI_window_release_received']=(a.out/'UI_DONE').exists();server.send_signal(signal.SIGTERM);returncode=server.wait(timeout=180);pump.join(timeout=10);result['server_lifecycle']={'returncode':returncode,'reaped':server.poll()is not None,'stdout_EOF':state['EOF'],'reader_joined':not pump.is_alive(),'reader_error':state['error']}
   events=[json.loads(x)for x in audit.read_text().splitlines()];closed=[x for x in events if x['event']=='engine_close'];dones=[x for x in events if x['event']=='engine_DONE'];sends=[x for x in events if x['event']=='engine_send'];result['engine_close_audit']=closed;result['native_DONE_lines']=[x['line']for x in dones];result['native_STOP_seen']=any(x.get('line')=='STOP'for x in sends);result['GENI_packets']=[x['packet']for x in sends if x.get('packet')];result['thinking62_observed']=bool(result['GENI_packets']) and all(x.get('reasoning_effort_observed')==[62] for x in sends if x.get('command') in ('GEN','GENI'));result['encode_completed']=[x for x in(a.out/'engine.log').read_text().splitlines()if x.startswith('ds4 vision encode: completed=1 ')]
   if not closed or not all(x.get('returncode')==0 and all(x.get(k)is True for k in ['stdin_closed','stdout_closed','reader_joined','reaped'])for x in closed):raise RuntimeError('normal engine QUIT/EOF/reap unproved')
   if Path('/proc',str(engine)).exists():raise RuntimeError('engine survived server shutdown')
   result['pass_']=result['UI_window_release_received'] and result['thinking62_observed'] and returncode==0 and state['EOF']and state['error']is None and result['native_STOP_seen']and any(x.split()[5]=='cancel'for x in result['native_DONE_lines'])and len(result['GENI_packets'])>=5 and len(result['encode_completed'])>=3
  except Exception as e:result['error']=repr(e)
  finally:
   for pid in [engine,server.pid if server else None]:
    if pid is None:continue
    live,zombies=group_members(pid)
    if live:
     result['pass_']=False
     try:os.killpg(pid,signal.SIGKILL)
     except ProcessLookupError:pass
   if server is not None:
    try:server.wait(timeout=8)
    except subprocess.TimeoutExpired:result.update(unreaped_server=True,pass_=False)
   if engine is not None:
    until=time.monotonic()+8
    while time.monotonic()<until:
     try:
      reaped,_=os.waitpid(engine,os.WNOHANG)
      if reaped==engine:break
     except ChildProcessError:
      if not Path('/proc',str(engine)).exists():break
     time.sleep(.05)
   if pump is not None:pump.join(timeout=10)
   if log is not None:log.close()
   result['residual_groups']={str(pid):group_members(pid)[0]for pid in [engine,server.pid if server else None]if pid is not None}
   if any(result['residual_groups'].values()):result['pass_']=False
   try:guards(g);payload_guard(gcfg['payload_stat_guard']);result['guard_post']=True
   except Exception as e:result.update(guard_post=False,guard_error=repr(e),pass_=False)
   try:result['resources']=critical(before,snapshot(os.getpid()));result['pass_']&=result['resources']['pass_']
   except Exception as e:result.update(resource_error=repr(e),pass_=False)
   result['elapsed_s']=time.monotonic()-start;save()
   if any(result['residual_groups'].values()):raise RuntimeError('owned resource descendants remain; inherited Dlock protects queue')
  return 0 if result['pass_']else 2
if __name__=='__main__':raise SystemExit(main())

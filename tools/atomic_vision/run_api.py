#!/usr/bin/env python3
"""Known-image API replay. Parent owns locked server/engine lifecycle; this starts no model."""
import argparse,base64,hashlib,json,os,re,time,urllib.request
from pathlib import Path
import sys
sys.path.insert(0,'/home/qwopqwop200/Strata-tuning/upstream-027/atomic-agent2-vision-v1/tools/atomic_vision')
from fixtures import QUESTION
from quality import grade
from resources import snapshot,critical
LOCK=Path('/home/qwopqwop200/Strata-tuning/.validation.lock');STOP=LOCK.parent/'upstream-027/STOP'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proc(pid):
 p=Path('/proc')/str(pid);stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
 return {'pid':pid,'starttime':int(tail[19]),'exe':os.readlink(p/'exe'),'argv':(p/'cmdline').read_bytes().split(b'\0')[:-1],'io':(p/'io').read_text(),'status':(p/'status').read_text(),'stat':stat}
def lease(pid):
 if STOP.exists():raise RuntimeError('STOP: no new request')
 p=Path('/proc')/str(pid);require=False
 for fd in (p/'fd').iterdir():
  try:
   if Path(os.readlink(fd))==LOCK and re.search(r'lock:\s+\d+: FLOCK\s+ADVISORY\s+WRITE', (p/'fdinfo'/fd.name).read_text()):require=True
  except (OSError,ValueError):pass
 if not require:raise RuntimeError('parent shared exclusive FLOCK ownership not visible; refusing server client')
def request(url,body,timeout):
 req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout)as response:return json.load(response)
def payload(model,case,image,stream=False,mixed=False):
 content=[{'type':'text','text':('This is a new picture; do not reuse the answer to an earlier picture. 'if mixed else'')+QUESTION},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}}]
 return {'model':model,'messages':[{'role':'user','content':content}],'max_tokens':1024,'temperature':0,'stream':stream,'enable_thinking':True}
def api_text(message):
 # Preserve API fields in the receipt. This reconstruction is not raw engine text.
 reasoning=message.get('reasoning_content',message.get('reasoning',''))or'';content=message.get('content')or''
 return reasoning+'</think>'+content if reasoning else content

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True);ap.add_argument('--model',required=True);ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--engine-pid',required=True,type=int);ap.add_argument('--lock-owner-pid',required=True,type=int);ap.add_argument('--binary-sha256',required=True);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--label',choices=['OLD_MODEL_VISION_PREPARATION','ATOMIC_MODEL_VISION_VALIDATION'],required=True);a=ap.parse_args()
 if not a.endpoint.startswith('http://127.0.0.1:'):raise ValueError('only private localhost endpoint')
 if a.out.exists():raise FileExistsError(a.out)
 a.out.mkdir(parents=True);manifest=json.loads(a.manifest.read_text());identity=proc(a.engine_pid);require_argv=[v.decode()for v in identity.pop('argv')]
 if sha(identity['exe'])!=a.binary_sha256:raise ValueError('actual engine binary changed')
 actual_env=dict(x.split('=',1)for x in (Path('/proc')/str(a.engine_pid)/'environ').read_text().split('\0')if '='in x)
 if actual_env.get('DS4_REQUIRE_VISION')!='1':raise ValueError('strict original vision contract not active')
 if '--experts'not in require_argv or require_argv[require_argv.index('--experts')+1]!='6':raise ValueError('full6 required')
 resource_before=snapshot(a.engine_pid);result={'resource_before':resource_before,'label':a.label,'AtomicPASS':False,'manifest_sha256':sha(a.manifest),'actual_engine':identity,'actual_argv':require_argv,'cases':[],'cancel':None,'pass_':False,'limits':['Parent owns server start/QUIT/EOF/reap under common lock; this client starts or kills no model.','Each record includes original API response; reconstructed reasoning text is not raw engine stream.','Not 1M or text/code acceptance.']}
 def receipt():
  (a.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 try:
  for name,mixed in [('red_square',False),('red_square',False),('blue_circle',True)]:
   lease(a.lock_owner_pid);before=proc(a.engine_pid);c=manifest['cases'][name];image=Path(c['image']).read_bytes()
   if hashlib.sha256(image).hexdigest()!=c['image_sha256']:raise ValueError('fixture changed')
   t=time.monotonic();r=request(a.endpoint,payload(a.model,c,image,mixed=mixed),240);after=proc(a.engine_pid);choice=r['choices'][0]
   text=api_text(choice['message']);quality=grade(text,c['expected'],choice['finish_reason']);same=before['starttime']==after['starttime']==identity['starttime'] and before['exe']==after['exe']==identity['exe'];row={'fixture':name,'mixed_text':mixed,'response':r,'quality':quality,'same_engine':same,'wall_s':time.monotonic()-t,'before':before,'after':after}
   for s in (row['before'],row['after']):s['argv']=[x.decode()for x in s['argv']]
   result['cases'].append(row);receipt()
   if not quality['pass_']or not same:raise RuntimeError('actual image response failed content or PID reuse')
  # Disconnect a real streamed image request, then require the same process to
  # accept another image. A transport close is not itself a certified STOP/drain.
  lease(a.lock_owner_pid);c=manifest['cases']['green_triangle'];body=payload(a.model,c,Path(c['image']).read_bytes(),stream=True);req=urllib.request.Request(a.endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});seen=0;start=time.monotonic()
  with urllib.request.urlopen(req,timeout=120)as response:
   for line in response:
    if line.startswith(b'data: ')and line.strip()!=b'data: [DONE]':
     event=json.loads(line[6:]);delta=(event.get('choices')or[{}])[0].get('delta',{})
     if delta.get('content')or delta.get('reasoning_content'):seen+=1
     if seen>=8:break
  result['cancel']={'stream_closed_after_nonempty_events':seen,'wall_s':time.monotonic()-start,'engine_STOP_confirmed':False,'limits':'Client disconnect only; followup success proves usable process, not exact cancellation token boundary.'};receipt()
  if seen<8:raise RuntimeError('stream ended before intended cancellation point')
  lease(a.lock_owner_pid);r=request(a.endpoint,payload(a.model,c,Path(c['image']).read_bytes(),mixed=True),240);q=r['choices'][0];quality=grade(api_text(q['message']),c['expected'],q['finish_reason']);after=proc(a.engine_pid);after['argv']=[x.decode()for x in after['argv']]
  result['followup']={'response':r,'quality':quality,'engine':after};resource_after=snapshot(a.engine_pid);result['resource_after']=resource_after;result['resources']=critical(resource_before,resource_after);result['pass_']=quality['pass_']and after['starttime']==identity['starttime']and after['exe']==identity['exe']and result['resources']['pass_'];result['atomic_vision_api_content_pass']=result['pass_']and a.label=='ATOMIC_MODEL_VISION_VALIDATION';result['full_atomic_acceptance_certified']=False;receipt();return 0 if result['pass_']else 2
 except Exception as e:result['error']=repr(e);receipt();return 2
if __name__=='__main__':raise SystemExit(main())

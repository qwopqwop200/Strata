"""Validation-only module-local seams: inherited Dlock, engine INFO/GENI/STOP/DONE and EOF receipts."""
import ctypes,hashlib,json,os,re,signal,struct,sys,threading,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from serve import server
sys.path.insert(0,str(R.parent/'atomic-agent2-vision-v1/tools/atomic_vision'))
from tokenizer_assets import load as load_tokenizer
tokenizer=load_tokenizer(os.environ['STRATA_VALIDATION_TOKENIZER'])
LOCK=Path('/home/qwopqwop200/Strata-tuning/.validation.lock')
fd=int(os.environ['STRATA_VALIDATION_LOCK_FD']);audit=Path(os.environ['STRATA_VALIDATION_AUDIT']);owner=os.fstat(fd);expected=LOCK.stat()
if (owner.st_dev,owner.st_ino)!=(expected.st_dev,expected.st_ino):raise ValueError('not inherited task Dlock')
mu=threading.Lock();count=0

def note(kind,**values):
 global count
 with mu:
  count+=1
  if count>256:raise RuntimeError('validation audit bound exceeded')
  with audit.open('a')as f:f.write(json.dumps(dict(event=kind,mono=time.monotonic(),**values))+'\n')

original_popen=server.popen

def locked_popen(what,args,role='engine',**kw):
 previous=kw.get('preexec_fn');parent=os.getpid();kw['pass_fds']=tuple(set(kw.get('pass_fds',()))|{fd})
 def guard():
  if previous:previous()
  if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
 kw['preexec_fn']=guard
 child=original_popen(what,args,role=role,**kw)
 if role=='engine':child.stdin=AuditedInput(child.stdin,child.pid)
 note('child_spawn',pid=child.pid,role=role,argv=list(args));return child
server.popen=locked_popen
original_init=server.StrataEngine.__init__
def init(self,*args,**kw):
 value=original_init(self,*args,**kw)
 if self.proc is not None:note('engine_ready',pid=self.proc.pid,info=self.info)
 return value
server.StrataEngine.__init__=init
def packet_meta(line):
 parts=line.split();path=next((p for p in parts[2:]if '='not in p),None)
 if path is None:raise ValueError('GENI packet path absent')
 data=Path(path).read_bytes()
 if len(data)>64*1024**2 or data[:4]!=b'DSI1':raise ValueError('validation expects bounded native DSI1 packet')
 count=struct.unpack('<i',data[4:8])[0];offset=8;images=[]
 if not 1<=count<=16:raise ValueError('validation image count bound')
 for _ in range(count):
  start,w,h=struct.unpack('<qii',data[offset:offset+16]);offset+=16
  if start<0 or w<=0 or h<=0 or w*h>64*1024**2:raise ValueError('DSI1 geometry invalid')
  pixels=data[offset:offset+w*h*3];offset+=w*h*3
  if len(pixels)!=w*h*3:raise ValueError('truncated DSI1 pixels')
  images.append(dict(start=start,width=w,height=h,rgb_sha256=hashlib.sha256(pixels).hexdigest()))
 if offset!=len(data):raise ValueError('DSI1 trailing bytes')
 return dict(path=path,sha256=hashlib.sha256(data).hexdigest(),images=images)
class AuditedInput:
 """Observe the actual pipe writes, including generate/STOP direct-write paths."""
 def __init__(self,stream,pid):self.stream=stream;self.pid=pid;self.pending=''
 def __getattr__(self,key):return getattr(self.stream,key)
 def write(self,text):
  written=self.stream.write(text)
  self.pending+=text[:written]
  if len(self.pending)>131072:raise RuntimeError('validation input audit bound exceeded')
  while '\n' in self.pending:
   line,self.pending=self.pending.split('\n',1)
   if line.startswith(('GENI ','GEN ','STOP','QUIT')):
    row=dict(pid=self.pid,line_sha256=hashlib.sha256(line.encode()).hexdigest(),write_returned=True)
    if line.startswith(('GENI ','GEN ')):
     ids=[int(x) for x in line.rsplit(' ',1)[1].split(',')]
     if len(ids)>4096:raise RuntimeError('short image API validation input bound')
     decoded=tokenizer.decode(ids)
     row.update(command=line.split()[0],input_ids=ids,input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),reasoning_effort_observed=[int(x) for x in re.findall(r'Reasoning Effort: (\d+) \(range 1-100,',decoded)],decoded_prompt_sha256=hashlib.sha256(decoded.encode()).hexdigest(),packet=packet_meta(line) if line.startswith('GENI ') else None)
    else:row.update(line=line,command=line)
    note('engine_send',**row)
  return written
original_done=server.StrataEngine._parse_done
def done(self,line):
 note('engine_DONE',pid=self.proc.pid,line=line);return original_done(self,line)
server.StrataEngine._parse_done=done
original_close=server.StrataEngine.close
def close(self):
 p=self.proc;pump=self.pump
 try:return original_close(self)
 finally:
  if p is not None:note('engine_close',pid=p.pid,returncode=p.poll(),stdin_closed=p.stdin.closed,stdout_closed=p.stdout.closed,reader_joined=pump is None or not pump.is_alive(),reaped=p.poll()is not None)
server.StrataEngine.close=close
if __name__=='__main__':raise SystemExit(server.main())

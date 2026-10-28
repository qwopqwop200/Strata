"""Prepare immutable Atomic backbone + original auxiliaries + retained Q4 draft.

Default prints a plan. --prepare requires downloader ALL_VERIFIED and shared lock.
No converter, tensor rewrite, duplicate backbone or Engram in-place mutation.
"""
from pathlib import Path
import argparse,json,hashlib,os,fcntl,subprocess,sys,signal,ctypes,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parent.parent
sys.path.insert(0,str(HERE));from retained_draft import attach_q4
ROOT=Path('/home/qwopqwop200/models/AtomicChat-DeepSeek-V4.1-IQ2_XXS')
LOCK=Path('/home/qwopqwop200/Strata-tuning/.validation.lock');STOP=REPO.parent/'STOP'
ATOMIC='5a2bf2d79e7c9021f6e36fcbbe3f41651308d922';ISTA='a419d74cd28d5b9f8c5e2b9077d11a609c17bc32'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def validate(root):
 m=json.loads((root/'download-manifest.json').read_text());s=json.loads((root/'download-status.json').read_text())
 if m.get('verified')is not True or s.get('stage')!='ALL_VERIFIED':raise ValueError('BLOCKED: downloader ALL_VERIFIED required')
 done={x['path']:x for x in s['completed']};seen=set();files=[]
 for x in m['files']:
  expected=ATOMIC if x['subdir']=='backbone' else ISTA
  repo='AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF'if x['subdir']=='backbone'else'ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF'
  if x['subdir']not in ['backbone','original']or x['repo']!=repo or x['revision']!=expected:raise ValueError('wrong pinned revision')
  p=root/x['subdir']/x['filename'];r=done.get(str(p));checksum=x.get('sha256')or x['git_blob']
  if not r or r.get('verified')is not True or r['checksum']!=checksum or r['size']!=x['size']or p.stat().st_size!=x['size']:raise ValueError('unverified/missing artifact '+str(p))
  seen.add(str(p));files.append(p)
 if len(seen)!=10 or len([x for x in m['files']if x['subdir']=='backbone'])!=4:raise ValueError('exact Atomic four shards +6original assets required')
 return m,s,files

def command(root,out):
 return [sys.executable,'-B',str(HERE/'pack_gguf.py'),'--model',str(root/'backbone'),'--out',str(out),'--engram',str(root/'original/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'),'--dspark-gguf',str(root/'retained-draft/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf'),'--dspark-noise-token','128799','--mmproj',str(root/'original/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf'),'--vision-router-bias',str(root/'original/exp_probs_b_vl.f32'),'--chat-template',str(root/'original/chat_template.jinja')]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--out',type=Path);ap.add_argument('--prepare',action='store_true');a=ap.parse_args();root=a.root.resolve();out=(a.out or root/'maya-ds4-q4').resolve()
 if not out.is_relative_to(root)or out==root or any(out.is_relative_to(root/n)for n in ['backbone','original','retained-draft']):raise ValueError('new dedicated pack path inside Atomic root required')
 plan=dict(root=str(root),out=str(out),command=command(root,out),target_conversion=False,Engram_in_place_mutation=False,requires='ALL_VERIFIED + retained manifest + frontend JoyAI alias',status='PLAN_ONLY')
 if not a.prepare:
  try:validate(root);plan['download_ready']=True
  except Exception as e:plan.update(download_ready=False,blocker=str(e))
  print(json.dumps(plan,indent=2));return 0
 with LOCK.open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if STOP.exists():raise ValueError('STOP after lock')
  if out.exists():raise ValueError('output already exists; preserve prior attempt and choose a new output path')
  m,s,inputs=validate(root);retained=root/'retained-draft';ret=json.loads((retained/'retention-manifest.json').read_text())
  inputs += [retained/'retention-manifest.json',retained/'historical-draft-quant.json',retained/'historical-pack/maya-ds4-q4/tensors.txt']+[Path(x['destination'])for x in ret['artifacts']]
  before={str(p):identity(p)for p in set(inputs)};source_before={str(p):sha(p)for p in [HERE/'pack_gguf.py',HERE/'retained_draft.py',HERE/'atomic_mapping.py',REPO/'tools/strata_tokenizer.py',Path(__file__).resolve()]};parent=os.getpid()
  def setup():
   if ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
  p=subprocess.Popen(command(root,out),pass_fds=(lock.fileno(),),start_new_session=True,preexec_fn=setup)
  try:rc=p.wait(timeout=120)
  except BaseException:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait(timeout=3);raise
  if rc:raise ValueError('pack tool failed exit '+str(rc))
  if STOP.exists():raise ValueError('STOP between pack and Q4 metadata attachment')
  proof=attach_q4(out,root/'backbone',retained)
  from pack_gguf import Gguf
  first=next(root/'backbone'/x['filename']for x in m['files']if x['subdir']=='backbone'and'-00001-'in x['filename'])
  tokenizer_kv={k:v for k,v in Gguf(first).kv.items()if k.startswith('tokenizer.')}
  tokenizer_source=dict(kind='atomic_embedded_gguf',repo='AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF',revision=ATOMIC,source_file=str(first),header_probe=str(REPO/'header-probes'/(first.name+'.header-prefix')),tokenizer_metadata_sha256=hashlib.sha256(json.dumps(tokenizer_kv,sort_keys=True,separators=(',',':')).encode()).hexdigest(),full_files_verified=True,chat_template_override=str(root/'original/chat_template.jinja'),chat_template_sha256=sha(root/'original/chat_template.jinja'))
  tokenizer_source['export_sha256']={n:sha(out/'tokenizer'/n)for n in ['vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja']}
  (out/'tokenizer/tokenizer-source.json').write_text(json.dumps(tokenizer_source,indent=2)+'\n')
  if before!={str(p):identity(p)for p in set(inputs)}or source_before!={p:sha(p)for p in source_before}:raise ValueError('immutable input/source stat identity drift')
  from atomic_mapping import verify_pack
  mapping=verify_pack(out,root,m)
  text=(out/'tensors.txt').read_text()
  if 'Abiray-'in text:raise ValueError('old model path leaked into new runtime dependency table')
  result=dict(schema=1,status='PACK_PREPARED_NOT_MODEL_VALIDATED',pack_root=str(out),model_root=str(root),actual_new_model_pack=True,files_verified=True,target_repo='AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF',target_revision=ATOMIC,auxiliary_revision=ISTA,target_shards=[dict(repo=x['repo'],path=str(root/x['subdir']/x['filename']),size=x['size'],sha256=x['sha256'],revision=x['revision'])for x in m['files']if x['subdir']=='backbone'],original_auxiliaries=[dict(x,path=str(root/x['subdir']/x['filename']))for x in m['files']if x['subdir']=='original'],tokenizer_source=tokenizer_source,mapping_proof=mapping,pack_files_sha256={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()},download_manifest_sha256=sha(root/'download-manifest.json'),download_status_sha256=sha(root/'download-status.json'),source_sha256=source_before,input_stat_prepost_equal=True,full_payload_hashes='Downloader remote checksums ALL_VERIFIED; no redundant 249GB hash in metadata preparation',retained_Q4=proof,target_weight_payload_modified=False,Engram_payload_modified=False,model_acceptance_pass=False,vision_pass=False,context1M_pass=False,tensor_table_sha256=sha(out/'tensors.txt'),ds4_sha256=sha(out/'ds4.txt'),requires_followup=['Atomic text/code/image/1M actual runtime','frontend/tokenizer JoyAI alias integrated','default launcher path readback','dependency closure before old deletion'])
  (out/'atomic-pack-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

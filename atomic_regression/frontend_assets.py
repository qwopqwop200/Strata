"""Atomic-specific, fail-closed tokenizer/template contract; no network/model calls."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from serve.frontend import ChatTemplate,literal_tags,mark_think_literals,unmark_think_literals
from serve import dsml
import strata_tokenizer as ST
ATOMIC_REPO='AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF'
ATOMIC_REV='5a2bf2d79e7c9021f6e36fcbbe3f41651308d922'
ORIGINAL_REPO='ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF'
ORIGINAL_REV='a419d74cd28d5b9f8c5e2b9077d11a609c17bc32'
MODEL=Path('/home/qwopqwop200/models/AtomicChat-DeepSeek-V4.1-IQ2_XXS')
TEMPLATE=MODEL/'original/chat_template.jinja'
TEMPLATE_GIT_BLOB='8127f8d9ceaf293b3d7bc75121d279f8887f8479'
TOKENIZER_FILES=('vocab.json','merges.txt','token_type.json','tokenizer.json')
IMAGE='<｜deepseek_image｜>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_sha(v):return hashlib.sha256(json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def pinned_template(path=TEMPLATE):
 p=Path(path);b=p.read_bytes();git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 if git!=TEMPLATE_GIT_BLOB:raise ValueError('original pinned ISTA template mismatch')
 selected,swapped=dsml.pick_template(p)
 if swapped or selected!=p:raise ValueError('frontend would replace original pinned template')
 return ChatTemplate(p)
def verify_export(root,metadata):
 """Compare exported files to actual Atomic tokenizer KV; no old-pack fallback."""
 root=Path(root);vocab=json.loads((root/'vocab.json').read_text());tokens=metadata['tokenizer.ggml.tokens'];merges=metadata['tokenizer.ggml.merges'];types=metadata['tokenizer.ggml.token_type'];cfg=json.loads((root/'tokenizer.json').read_text())
 if len(set(tokens))!=len(tokens) or vocab!={x:i for i,x in enumerate(tokens)}:raise ValueError('vocab differs from Atomic header')
 if (root/'merges.txt').read_text().split('\n')!=merges:raise ValueError('merge ranks differ from Atomic header')
 if json.loads((root/'token_type.json').read_text())!=types or len(types)!=len(tokens):raise ValueError('token types differ from Atomic header')
 pre=metadata.get('tokenizer.ggml.pre');special={k:int(v) for k,v in metadata.items() if k.startswith('tokenizer.ggml.') and k.endswith('_token_id')}
 if metadata.get('tokenizer.ggml.model')!='gpt2' or not isinstance(pre,str) or not (pre.startswith('deepseek') or pre=='joyai-llm'):raise ValueError('unexpected Atomic tokenizer pre/model')
 if cfg.get('pre')!=pre or cfg.get('vocab_size')!=len(tokens) or cfg.get('special_ids')!=special:raise ValueError('exported tokenizer config differs')
 tok=ST.Tokenizer(tokens,merges,types,pre,special)
 for token in ['<｜begin▁of▁sentence｜>','<｜User｜>','<｜Assistant｜>','<think>','</think>',IMAGE]:
  if token not in tok.ids or len(tok.encode(token,parse_special=True))!=1:raise ValueError('missing/split required special '+token)
 return tok

def make_contract(tokenizer_root,metadata,provenance,out):
 """Called after track1 export; partial header proofs remain CPU-only."""
 if provenance.get('kind')!='atomic_embedded_gguf' or provenance.get('repo')!=ATOMIC_REPO or provenance.get('revision')!=ATOMIC_REV:raise ValueError('tokenizer provenance must be pinned Atomic header')
 if provenance.get('tokenizer_metadata_sha256')!=canonical_sha(metadata):raise ValueError('Atomic header metadata hash mismatch')
 verify_export(tokenizer_root,metadata);pinned_template()
 root=Path(tokenizer_root).resolve();files={str(root/n):sha(root/n) for n in TOKENIZER_FILES};files[str(TEMPLATE)]=sha(TEMPLATE)
 c={'schema':1,'target_repo':ATOMIC_REPO,'target_revision':ATOMIC_REV,'original_repo':ORIGINAL_REPO,'original_revision':ORIGINAL_REV,'tokenizer_root':str(root),'tokenizer_metadata':metadata,'tokenizer_source':provenance,'template':str(TEMPLATE),'template_git_blob':TEMPLATE_GIT_BLOB,'sha256':files,'thinking_effort':62,'experts':6,'dense_draft':'Q4','model_ready':False,'limits':'Tokenizer/header provenance only. Model execution requires ALL_VERIFIED download plus actual pack/build/model guards. Images require actual vision span expansion, not one placeholder per image.'}
 Path(out).write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');return c
class AtomicFrontend:
 def __init__(self,contract):
  self.contract=json.loads(Path(contract).read_text())
  c=self.contract
  if c['target_repo']!=ATOMIC_REPO or c['target_revision']!=ATOMIC_REV or c['thinking_effort']!=62 or c['experts']!=6 or c['dense_draft']!='Q4':raise ValueError('Atomic original model contract changed')
  for p,h in c['sha256'].items():
   if sha(p)!=h:raise ValueError('frontend asset changed '+p)
  source=c['tokenizer_source'];metadata=c['tokenizer_metadata']
  if source['kind']!='atomic_embedded_gguf' or source['repo']!=ATOMIC_REPO or source['revision']!=ATOMIC_REV or source['tokenizer_metadata_sha256']!=canonical_sha(metadata):raise ValueError('Atomic tokenizer source not bound')
  self.tok=verify_export(c['tokenizer_root'],metadata);self.template=pinned_template(c['template'])
 def render_messages(self,messages,tools=None):
  tags=literal_tags(self.tok.control_tokens);marked,marked_tools,changed=mark_think_literals(messages,tools,tags)
  text=self.template.render(marked,tools=marked_tools,thinking_mode='thinking',reasoning_effort=62,add_generation_prompt=True)
  plain=()
  if changed:text,plain=unmark_think_literals(text,tags)
  ids=self.tok.encode(text,parse_special=True,plain=plain)
  image_id=self.tok.ids[IMAGE];positions=[i for i,v in enumerate(ids) if v==image_id]
  return {'text':text,'ids':ids,'input_tokens':len(ids),'input_ids_sha256':canonical_sha(ids),'rendered_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'asset_contract_sha256':canonical_sha(self.contract),'reasoning_effort':62,'image_placeholder_positions':positions,'vision_expansion_pending':bool(positions),'model_ready':False}
 def encode_plain(self,text):
  return self.tok.encode(text,parse_special=False)
 def decode(self,ids):
  return self.tok.decode(ids)
 def assert_download_complete(self,status_path=MODEL/'download-status.json',manifest_path=MODEL/'download-manifest.json'):
  status=json.loads(Path(status_path).read_text());manifest=json.loads(Path(manifest_path).read_text())
  if status.get('stage')!='ALL_VERIFIED' or manifest.get('verified') is not True:raise ValueError('new full-model files not ALL_VERIFIED')
  rows={r['path']:r for r in status['completed']};backbone=[]
  for f in manifest['files']:
   p=MODEL/f['subdir']/f['filename'];row=rows.get(str(p));expected=f.get('sha256') or f['git_blob']
   if not row or row.get('verified') is not True or row['checksum']!=expected or p.stat().st_size!=f['size']:raise ValueError('missing verified download '+str(p))
   if f['subdir']=='backbone':
    if f['repo']!=ATOMIC_REPO or f['revision']!=ATOMIC_REV:raise ValueError('backbone pin changed')
    backbone.append(p)
  if len(backbone)!=4:raise ValueError('Atomic four-shard target required')
  return {'download_complete':True,'limits':'Recorded service checksum receipts verified, not duplicate large-file hashing; actual runtime/pack integrity is a separate guard.'}

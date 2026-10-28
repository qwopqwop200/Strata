"""CPU-only capacity formulas and input-completion validation; no engine launch."""
import json,math,pathlib,re
MAX_TOTAL=1048576
MIN_REAL_INPUT=1000000
DEFAULT_INPUT=1000000
DEFAULT_OUTPUT_RESERVE=4096

def read_geometry(path):
 text=pathlib.Path(path).read_text()
 if text.lstrip().startswith('{'):
  data=json.loads(text);g=data.get('geometry',data)
  return {k:','.join(map(str,v))if isinstance(v,list)else str(v)for k,v in g.items()}
 result={}
 for line in text.splitlines():
  parts=line.split(None,1)
  if len(parts)==2:result[parts[0]]=parts[1]
 return result

def estimate(geometry,context,chunk=8192):
 maximum=int(geometry['max_position_embeddings']);nl=int(geometry['num_hidden_layers']);hd=int(geometry['head_dim']);ih=int(geometry['index_head_dim']);block=int(geometry['candidate_block_size'])
 ratios=list(map(int,geometry['compress_ratios'].split(',')));sources=list(map(int,geometry['kv_source_layer_ids'].split(',')))
 if not 0<context<=maximum or context>2**31-65:raise ValueError('metadata position limit/int index bound')
 if nl<=0 or len(ratios)<nl or hd<=0 or ih<=0 or hd%32 or ih%32 or block<=0 or not sources or len(set(sources))!=len(sources):raise ValueError('geometry')
 if any(not 0<=i<nl or ratios[i]<=0 for i in sources):raise ValueError('compressed source')
 caps={str(i):(context+ratios[i]-1)//ratios[i]+1 for i in sources};span=context+64
 rb=lambda x:(x+15)//16*16
 slab=max(32,min(128,((64<<20)//(span*4))//32*32))
 return dict(metadata_max=maximum,context=context,source_caps=caps,source_rows=sum(caps.values()),source_cache_bytes=sum(caps.values())*(rb(hd//2+hd//16)+rb(ih//2+ih//32)),decode_score_bytes=16*span*4,prefill_score_bytes=slab*span*4 if chunk>0 else 0,index_slab_rows=slab if chunk>0 else 0,bitmap_stride=span//block+64,accounting='partial_single_device: excludes weights/Engram/sliding/recurrent/chunk/BLAS/allocator/host memory; not OOM admission')

def fingerprint(ids):
 h=14695981039346656037
 for token in ids:
  if type(token)is not int or not 0<=token<2**31:raise ValueError('nonnegative int32 IDs required')
  for b in int(token).to_bytes(4,'little'):h=((h^b)*1099511628211)&((1<<64)-1)
 return f'{h:016x}'

def typed_equal(a,b):
 if type(a)is not type(b):return False
 if isinstance(b,dict):return a.keys()==b.keys()and all(typed_equal(a[k],v)for k,v in b.items())
 if isinstance(b,list):return len(a)==len(b)and all(typed_equal(x,y)for x,y in zip(a,b))
 return a==b

def validate_completion(request,fixture,stderr,stdout):
 ids=fixture['ids'];expected=fixture['expected'];n=len(ids);cap=fixture['max_new'];context=fixture['context'];d=request['done'].split()
 checks=dict(real_1m_input=n>=MIN_REAL_INPUT,input_plus_output_within_metadata=n+cap<=context<=MAX_TOTAL,strict_DONE11=False,input_marker=False,full_progress=False,meaningful_recall=False)
 try:checks['strict_DONE11']=len(d)==11 and d[0]=='DONE'and int(d[1])==len(request['tokens'])and 0<int(d[1])<=cap and int(d[2])==n and d[5]=='stop'and 0<=int(d[6])<=int(d[7])and d[8:]==['0','0','0']and all(math.isfinite(float(v))and float(v)>0 for v in d[3:5])
 except(ValueError,TypeError):pass
 pattern=r'ds4 context input: requested=1 input_tokens=(\d+) fresh_from=(\d+) prompt_read=(\d+) engine_position=(\d+) context=(\d+) reuse_enabled=(\d+) input_fnv64=([0-9a-f]{16}) full_input_position_match=(\d+)'
 lines=[l for l in stderr.splitlines()if l.startswith('ds4 context input:')];m=re.fullmatch(pattern,lines[0])if len(lines)==1 else None
 checks['input_marker']=bool(m and [int(m[i])for i in (1,2,3,4,5,6,8)]==[n,0,n,n,context,0,1]and m[7]==fixture['input_fnv64'])
 progress=[tuple(map(int,m))for m in re.findall(r'^PP (\d+) (\d+) [^\s]+ [^\s]+ (\d+)$',stdout,re.M)]
 checks['full_progress']=bool(progress)and progress[-1]==(n,n,0)and all(total==n and reused==0 and 0<read<=n for read,total,reused in progress)and all(a[0]<=b[0]for a,b in zip(progress,progress[1:]))
 try:
  answer=request['text'].split('</think>',1)[1].split('<｜end▁of▁sentence｜>')[0].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip();answer=json.loads(answer)
  checks['meaningful_recall']=type(answer)is dict and typed_equal(answer,expected)
 except(ValueError,IndexError,TypeError):pass
 return dict(pass_=all(checks.values()),checks=checks,limits='FNV64 is a noncryptographic all-ID transport fingerprint, not proof every layer attended every token. Full position/progress/fresh0 and far-position meaningful recall are independent gates; actual model/hash/PID/cgroup/lifecycle guards are separate mandatory driver gates.')

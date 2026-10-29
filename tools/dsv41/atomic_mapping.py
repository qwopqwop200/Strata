"""Prove pack target records are unchanged byte views of Atomic GGUF tensors."""
from pathlib import Path
import math,re,hashlib,json
from pack_gguf import Gguf,TYPES,LAYER,GLOBAL
from retained_draft import read_table
METADATA={'engram_token_map.weight','engram_primes.weight','engram_offsets.weight','engram_multipliers.weight'}
def expected_backbone(gs,filesizes,groups):
    expected={};source_count=0
    for g in gs:
        for name,typ,dims,off in g.tensors:
            if name in METADATA:
                if typ not in (26,27):raise ValueError('integer metadata type '+name)
                continue
            if typ not in TYPES:raise ValueError('unrecognized Atomic weight type '+name)
            dt,per,bpb=TYPES[typ];nbytes=math.prod(dims)//per*bpb;base=g.data_start+off;shape=list(reversed(dims))
            if dims[0]%per or base<g.data_start or base+nbytes>filesizes[str(g.path)]:raise ValueError('invalid Atomic tensor range '+name)
            m=re.fullmatch(r'blk\.(\d+)\.(.+)',name)
            if m and m[2]in LAYER:ck='layers.'+m[1]+'.'+LAYER[m[2]]
            elif name in GLOBAL:ck=GLOBAL[name]
            else:raise ValueError('unmapped Atomic weight '+name)
            source_count+=1
            if ck.endswith('.attn.wo_a.slice'):
                if len(shape)==3 and shape[0]==groups:shape=[shape[0]*shape[1],shape[2]]
                if len(shape)!=2 or shape[0]%groups or nbytes%groups:raise ValueError('wo_a group geometry')
                rows=[(ck+f'.{i}.weight',base+i*(nbytes//groups),nbytes//groups,[shape[0]//groups,shape[1]])for i in range(groups)]
            elif '.ffn.experts.'in ck:
                count=shape[0]
                if nbytes%count:raise ValueError('expert tensor partition')
                pre,role=ck.rsplit('.',1);rows=[(f'{pre}.{i}.{role}.weight',base+i*(nbytes//count),nbytes//count,shape[1:])for i in range(count)]
            else:rows=[(ck,base,nbytes,shape)]
            for n,o,b,sh in rows:
                if n in expected:raise ValueError('duplicate expected target '+n)
                expected[n]=[str(g.path.resolve()),dt,o,b,sh]
    return expected,source_count

def verify_pack(pack,root,manifest):
    root,pack=Path(root).resolve(),Path(pack).resolve();model=root/'backbone'
    shard_items=[x for x in manifest['files']if x['subdir']=='backbone'];paths=[model/x['filename']for x in shard_items];gs=[Gguf(p)for p in paths];sizes={str(p):x['size']for p,x in zip(paths,shard_items)}
    kv=next(g.kv for g in gs if g.kv.get('general.architecture')=='deepseek41');groups=int(kv['deepseek41.attention.output_group_count'])
    expected,count=expected_backbone(gs,sizes,groups);files,table=read_table(pack/'tensors.txt');actual={};closure=[]
    known={str(root/x['subdir']/x['filename']):x for x in manifest['files']};ret=json.loads((root/'retained-draft/retention-manifest.json').read_text());known.update({x['destination']:x for x in ret['artifacts']})
    for i,name in files.items():
        p=(model/name).resolve()
        if not p.is_relative_to(root)or str(p)not in known:raise ValueError('dependency outside verified new-root closure '+str(p))
        if p.stat().st_size!=known[str(p)].get('size'):raise ValueError('dependency size mismatch '+str(p))
        st=p.stat();closure.append(dict(file=i,path=str(p),size=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,checksum=known[str(p)].get('sha256')or known[str(p)].get('git_blob'),checksum_kind='sha256'if known[str(p)].get('sha256')else'git_blob_sha1',verified_by='download ALL_VERIFIED or retained parent manifest'))
    target_paths={str(p.resolve())for p in paths}
    for n,t in table.items():
        p=(model/files[int(t[3])]).resolve();off,nb=int(t[4]),int(t[5]);shape=list(map(int,t[6:]));dt=t[2]
        if off<0 or nb<=0 or off+nb>p.stat().st_size:raise ValueError('pack tensor out-of-bounds '+n)
        if str(p)in target_paths:actual[n]=[str(p),dt,off,nb,shape]
        elif n in expected:raise ValueError('Atomic target silently replaced by auxiliary '+n)
    if actual!=expected:raise ValueError('Atomic dtype/offset/shape/name views differ; missing='+str(sorted(expected.keys()-actual.keys())[:4]))
    return dict(pass_=True,target_source_tensor_count=count,target_runtime_tensor_count=len(expected),target_tensor_offsets_verified=True,target_dtype_preserved=True,target_tensor_byte_views_exact=True,stable_identity_fields=['device','inode','size','mtime_ns'],ctime_scope='advisory only; unlinking old hardlink aliases changes ctime without payload change',target_mapping_sha256=hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest(),mapped_files=closure,limits='Header+mapping proof over downloader whole-file checksum verified inputs; no redundant per-tensor payload rehash.')

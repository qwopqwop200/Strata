"""Metadata-only OLD_MODEL_VISION_PREPARATION; never an Atomic model pass."""
from pathlib import Path
import json,hashlib,os,importlib.util,types,fcntl,shutil
R=Path(__file__).resolve().parent;U=R.parent
OLD=Path('/home/qwopqwop200/models/Abiray-DeepSeek-V4.1-IQ2_XXS/maya-ds4-q4')
NEW=Path('/home/qwopqwop200/models/AtomicChat-DeepSeek-V4.1-IQ2_XXS')
OUT=R/'validation-atomic-pack/old-vision-aux-pack-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('pack_source',R/'tools/dsv41/pack_gguf.py');pg=importlib.util.module_from_spec(spec);spec.loader.exec_module(pg)
spec=importlib.util.spec_from_file_location('table_source',R/'tools/dsv41/retained_draft.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
def main():
 with (U.parent/'.validation.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (U/'STOP').exists():raise ValueError('STOP after lock')
  if OUT.exists():raise ValueError('preserve existing output; do not overwrite')
  status=json.loads((NEW/'download-status.json').read_text());done={Path(x['path']).name:x for x in status['completed']}
  for name in ['mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf','exp_probs_b_vl.f32']:
   if done.get(name,{}).get('verified')is not True:raise ValueError('verified original auxiliary missing '+name)
  cfg=dict(line.split(' ',1)for line in (OLD/'ds4.txt').read_text().splitlines()if ' 'in line);model=Path(cfg['model_dir'])
  fm,tm=rt.read_table(OLD/'tensors.txt');files=[fm[i]for i in range(len(fm))]
  tensors=[(f[1],f[2],int(f[3]),int(f[4]),int(f[5]),list(map(int,f[6:])))for f in tm.values()]
  vocab=json.loads((OLD/'tokenizer/vocab.json').read_text());tokens=['']*len(vocab)
  for token,i in vocab.items():tokens[i]=token
  kv={'tokenizer.ggml.tokens':tokens};rel=lambda p:os.path.relpath(p,model).replace(os.sep,'/')
  src=(R/'tools/dsv41/pack_gguf.py').read_text();start=src.index('    # the vision tower from unsloth\'s mmproj');end=src.index('    # trellis experts from',start)
  import textwrap
  env=dict(Gguf=pg.Gguf,TYPES=pg.TYPES,pathlib=__import__('pathlib'),re=__import__('re'),a=types.SimpleNamespace(mmproj=str(NEW/'original/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf')),model=model,files=files,tensors=tensors,kv=kv,rel=rel,find_mmproj=pg.find_mmproj)
  exec(compile(textwrap.dedent(src[start:end]),'actual_pack_mmproj_mapping','exec'),env)
  mm_geo=env['mm_geo'];bias=pg.add_vision_router_bias(files,tensors,model,NEW/'original/exp_probs_b_vl.f32',40,384)
  for n,dt,fi,off,nb,shape in tensors[:len(tm)]:
   f=tm[n]
   if [n,dt,str(fi),str(off),str(nb),*map(str,shape)]!=f[1:]:raise ValueError('old target mapping changed '+n)
  if any(k in cfg for k in mm_geo):raise ValueError('old pack already had vision geometry; do not replace')
  OUT.mkdir(parents=True)
  (OUT/'ds4.txt').write_text((OLD/'ds4.txt').read_text()+''.join(f'{k} {v}\n'for k,v in mm_geo.items()))
  extra=''.join(f'file {i} {files[i]}\n'for i in range(len(fm),len(files)))+''.join(f't {n} {dt} {fi} {off} {nb} '+ ' '.join(map(str,shape))+'\n'for n,dt,fi,off,nb,shape in tensors[len(tm):])
  (OUT/'tensors.txt').write_text((OLD/'tensors.txt').read_text()+extra)
  shutil.copyfile(OLD/'engram.bin',OUT/'engram.bin');shutil.copytree(OLD/'tokenizer',OUT/'tokenizer')
  # Old tokenizer is retained exactly; original new template is a separately
  # bound frontend asset, not silently installed into this historical pack.
  proof=dict(schema=1,status='OLD_MODEL_VISION_PREPARATION',actual_new_model_pack=False,Atomic_model_validated=False,old_pack=str(OLD),old_model_dir=str(model),old_target_mapping_preserved=True,old_target_tensor_count=len(tm),new_aux_tensor_count=len(tensors)-len(tm),original_mmproj=done['mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf'],bias=bias,vision_geometry=mm_geo,source_sha256={str(p):sha(p)for p in [Path(__file__),R/'tools/dsv41/pack_gguf.py',R/'tools/dsv41/retained_draft.py']},pack_files_sha256={str(p.relative_to(OUT)):sha(p)for p in OUT.rglob('*')if p.is_file()},original_pack_files_unchanged=True,weights_copied_or_modified=False,limits='Metadata-only OLD Abiray target + original image tower/bias; cannot count as Atomic text/vision/1M validation. Actual image/CUDA/runtime execution not done here.')
  (OUT/'OLD-VISION-PREPARATION.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()

from pathlib import Path
import importlib.util,json,hashlib
R=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('actual_frontend',R/'tools/strata_tokenizer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
kv=json.loads((R/'header-probes/TOKENIZER-KV.json').read_text());tokens=kv['tokenizer.ggml.tokens'];merges=kv['tokenizer.ggml.merges'];types=kv['tokenizer.ggml.token_type']
pack_spec=importlib.util.spec_from_file_location('actual_pack_export',R/'tools/dsv41/pack_gguf.py');pack=importlib.util.module_from_spec(pack_spec);pack_spec.loader.exec_module(pack)
export=R/'validation-atomic-pack/actual-tokenizer-v2';pack.write_tokenizer(export,kv,kv['tokenizer.chat_template']);cfg=json.loads((export/'tokenizer.json').read_text())
raw_special={k:int(v)for k,v in kv.items()if k.startswith('tokenizer.ggml.')and k.endswith('_token_id')}
assert cfg['gguf_special_ids']==raw_special and cfg['vocab_size']==len(tokens)
assert cfg['special_ids']=={'bos':0,'eos':1,'pad':1}
a=m.Tokenizer(tokens,merges,types,pre=cfg['pre'],special_ids=cfg['special_ids']);b=m.Tokenizer(tokens,merges,types,pre='deepseek3',special_ids=cfg['special_ids'])
assert a.special_ids['eos']==kv['tokenizer.ggml.eos_token_id'] and a.special_ids['pad']==kv['tokenizer.ggml.padding_token_id']
corpus=['','Hello, world!','한국어 테스트입니다.','你好，世界','😀🚀 e\u0301','  a\n\n\tb  ','12345678901234567890','<｜deepseek_image｜> image','def f(x):\n    return x + 1\n','\r\n\x00control','A/B/C: one-two_three','الفصحى']
rows=[]
for text in corpus:
 x,y=a.encode(text),b.encode(text);assert x==y;assert a.decode(x)==text;rows.append(dict(text=text,token_count=len(x),same_ids=True,roundtrip=True))
out=dict(pass_=True,scope='actualAtomic embedded metadata+released JoyAI alias CPU only',cases=rows,pre_preserved='joyai-llm',runtime_compact_specials=a.special_ids,raw_GGUF_specials=raw_special,no_missing_eos_or_pad_default=True,tokenizer_source_sha256=hashlib.sha256((R/'tools/strata_tokenizer.py').read_bytes()).hexdigest(),full_model_files_verified=False,models=0)
(R/'validation-atomic-pack/ACTUAL-TOKENIZER-CPU.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('PASS actualAtomic JoyAI/Deepseek3 sameIDs +roundtrip',len(rows))

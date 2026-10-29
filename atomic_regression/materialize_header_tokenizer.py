"""CPU-only export of already fetched Atomic header KV; never writes a model pack."""
import json
from pathlib import Path
from atomic_regression.frontend_assets import ST,make_contract
ROOT=Path(__file__).resolve().parents[1];PROBE=ROOT.parent/'atomic-agent1-pack-v1/header-probes'
def materialize(out):
 out=Path(out);out.mkdir(exist_ok=False,parents=True);md=json.loads((PROBE/'TOKENIZER-KV.json').read_text());source=json.loads((PROBE/'TOKENIZER-SOURCE.json').read_text());tokens=md['tokenizer.ggml.tokens'];special={k:int(v) for k,v in md.items() if k.startswith('tokenizer.ggml.') and k.endswith('_token_id')}
 (out/'vocab.json').write_text(json.dumps({t:i for i,t in enumerate(tokens)},ensure_ascii=False))
 (out/'merges.txt').write_text('\n'.join(md['tokenizer.ggml.merges']))
 (out/'token_type.json').write_text(json.dumps(md['tokenizer.ggml.token_type']))
 (out/'tokenizer.json').write_text(json.dumps({'model':'gpt2','pre':md['tokenizer.ggml.pre'],'vocab_size':len(tokens),'n_merges':len(md['tokenizer.ggml.merges']),'special_ids':special,'add_bos_token':md['tokenizer.ggml.add_bos_token'],'add_eos_token':md['tokenizer.ggml.add_eos_token'],'pre_pattern':list(ST.DEEPSEEK3_PATTERNS),'pre_pattern_source':'pinned llama3cf03257 src/llama-vocab.cpp319..327 shared JoyAI/DeepSeek3 sequence'},indent=2))
 return make_contract(out,md,source,out.parent/'HEADER-ONLY-ASSET-CONTRACT.json')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();c=materialize(a.out);print(json.dumps({'model_ready':c['model_ready'],'tokenizer_root':c['tokenizer_root'],'source':c['tokenizer_source']}))

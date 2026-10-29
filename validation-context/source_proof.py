from pathlib import Path
import hashlib,json,re,subprocess
R=Path(__file__).resolve().parents[1];BASE='b761fe96147349a23825f442db7b33a21dd45278'
sha=lambda x:hashlib.sha256(x).hexdigest()
proof={}
for rel in ('src/dsv41/ds4_model.cu','src/dsv41/ds4_main.cpp'):
 old=subprocess.check_output(['git','show',BASE+':'+rel],cwd=R);text=(R/rel).read_text();text=text.replace('#include "ds4_context_capacity.hpp"\n','')
 if rel.endswith('ds4_model.cu'):
  text=text.replace('    int64_t max_positions = 0;\n','').replace('        max_positions = (int64_t) p.num("max_position_embeddings", 0);\n','');text=re.sub(r'        // ATOMIC_CONTEXT_CAPACITY_BEGIN\n.*?        // ATOMIC_CONTEXT_CAPACITY_END\n','',text,flags=re.S)
 else:text=re.sub(r'        // ATOMIC_CONTEXT_INPUT_BEGIN\n.*?        // ATOMIC_CONTEXT_INPUT_END\n','',text,flags=re.S)
 assert text.encode()==old,rel
 proof[rel]=dict(base_sha256=sha(old),actual_sha256=sha((R/rel).read_bytes()),inverse_exact=True)
print(json.dumps(dict(status='PASS',source_inverse=proof,runtime_contract='strict1 optin capacity/input guard; no math/layout/YOCO/dispatch changes',CUDA_GPU_model_runs=0),indent=2))

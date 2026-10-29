from pathlib import Path
import json,re,hashlib,subprocess
D=Path(__file__).resolve().parent;R=D.parent
s=(R/'src/dsv41/ds4_kernels.cu').read_text();base='1d86877331c4632abe1c0282ce65b96b4445bd92'
old=subprocess.check_output(['git','show',base+':src/dsv41/ds4_kernels.cu'],cwd=R,text=True)
inv=re.sub(r'// PREFILL_ROUTER_REUSE(?:_DISPATCH)?_BEGIN\n.*?// PREFILL_ROUTER_REUSE(?:_DISPATCH)?_END\n\n?', '',s,flags=re.S)
assert inv==old
warp=s[s.index('__device__ __forceinline__ float warp_sum('):s.index('__device__ __forceinline__ float warp_max(')]
kernels=s[s.index('__global__ void __launch_bounds__(kThreads) gemv_f32_kernel('):s.index('// Direct kernel parameters, not a device-indexed by-value array.')]
wrapper=s[s.index('void gemv_f32(const float*'):s.index('void gemv_f32_router_static(')]
(D/'component.cu').write_text('#include "ds4_launch.h"\n#include <cstring>\n#include <mutex>\n#include <cstdint>\nnamespace strata::ds4 {\nconstexpr int kThreads=256;\nstatic int cdiv(int a,int b){return (a+b-1)/b;}\n'+warp+kernels+wrapper+'\n}\n#include "main.inc"\n')
pack=Path('/home/qwopqwop200/models/Abiray-DeepSeek-V4.1-IQ2_XXS/maya-ds4-q4');lines=(pack/'tensors.txt').read_text().splitlines();files={int(x.split()[1]):str(pack.parent/' '.join(x.split()[2:])) for x in lines if x.startswith('file ')}
t=next(x.split() for x in lines if x.startswith('t layers.0.ffn.gate.weight '));assert t[2]=='BF16' and t[6:]==['384','5120']
c=dict(name=t[1],path=files[int(t[3])],offset=int(t[4]),bytes=int(t[5]),n=384,k=5120)
(D/'PAYLOAD.json').write_text(json.dumps([c],indent=2)+'\n')
(D/'case.inc').write_text('static const char* payload_path='+json.dumps(c['path'])+';\nstatic const size_t payload_offset='+str(c['offset'])+'ull;\n')
(D/'SOURCE-PROOF.json').write_text(json.dumps(dict(base=base,whole_OFF_inverse=True,original_kernel_unchanged=True,actual_shape=c,actual_prefill_callsite='ds4_model.cu:8295 gemv_f32(l.gate_w,c.x,l.n_exp,D,T,c.glog,1024,s)',arithmetic='four explicit fmaf per float4 and identical warp XOR reduction for each token',limits='T>16 and <=8192 only; original static/decode untouched'),indent=2)+'\n')
print('PASS source inverse and readonly metadata',c)

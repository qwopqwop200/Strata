#!/usr/bin/env python3
"""Actual tap + seed input-loop CPU extraction. CUDA calls are queued doubles, not GPU proof."""
import argparse, hashlib, json, pathlib, subprocess, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
s=(ROOT/'src/dsv41/ds4_model.cu').read_text()
def body(name):
    start=s.index('    bool '+name+'('); opening=s.index('{',start); depth=1; end=opening+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[start:end]
tap=body('tap'); seed=body('seed_drafter'); seed=seed[:seed.index('        CK(cudaEventRecord(ev_seed, d.st));')]+'        return true;\n    }'
verify=body('verify');forward=body('forward');serial=body('forward_serial');step=body('step')
assert verify.index('float* const taps = decode_taps(T)') < verify.index('forward(tk, T, T, err, 0, taps)') < verify.index('if (!ok) return false') < verify.index('rollback(p0, T, m + 1)') < verify.index('seed_drafter(p0, m + 1, 0, err, taps)')
assert 'forward(&token, 1, want_logits ? 1 : 0, err, -1, decode_taps(1))' in step
assert 'forward_serial(p0, T, err, device_taps)' in forward
assert 'seed_drafter(p0, seed_rows < 0 ? T : seed_rows, 0, err, device_taps)' in forward
assert 'tap(devs[(size_t) l.dev], il, T, err, device_taps)' in serial
assert 'forward(tokens + a, T, a + T == n ? logits_rows : 0, err)' in s
assert s.count('h_taps') == 10, 'Reaudit all host tap consumers if count changes'
assert 'std::strcmp(std::getenv("DS4_DSPARK_DEVICE_TAPS"), "1") == 0' in s
base=subprocess.check_output(['git','show','833b226e:src/dsv41/ds4_model.cu'],cwd=ROOT,text=True)
# The entire sampled/greedy verdict, RNG, rollback and publication body is byte-identical.
old=base[base.index('        if (temperature <= 0.0f) {',base.index('    bool verify(')):base.index('    bool step(')]
new=s[s.index('        if (temperature <= 0.0f) {',s.index('    bool verify(')):s.index('    float* decode_taps(')]
assert new.replace('seed_drafter(p0, m + 1, 0, err, taps)','seed_drafter(p0, m + 1, 0, err)') == old
start=s.index('    float* decode_taps('); end=s.index('    bool step(',start)
helper=s[start:end]
stub=(ROOT/'tests/dsv41/device_taps_stub.inc').read_text()
code=stub.replace('// ACTUAL_TAP',tap).replace('// ACTUAL_SEED',seed).replace('// ACTUAL_ELIGIBILITY',helper)
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'extracted.cpp').write_text(code)
r={'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'extracted_sha256':hashlib.sha256(code.encode()).hexdigest(),'source_checks_pass':True,'cpu_double_only':True,'model_runs':0,'cuda_runs':0}
if not a.prepare_only:
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(out/'extracted.cpp'),'-o',str(out/'test')],check=True,timeout=25)
    subprocess.run([str(out/'test')],check=True,timeout=15)
    r['extraction_pass']=True
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

"""Metadata-only closure; model/binary large hashes are verified under runner's Dlock."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];V=R/'validation-ui-v2';U=R.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 old=json.loads((U/'atomic-agent2-api-v1/validation-api-old/READY.json').read_text());guards=dict(old['sha256'])
 files=subprocess.check_output(['git','-C',str(R),'ls-files'],text=True).splitlines()
 for n in files:
  p=R/n
  if p.is_file() and (n.startswith(('serve/','tools/atomic_vision/','validation-ui-v2/'))):guards[str(p)]=sha(p)
 for n in ['CPU-attempt1.log','CPU.log','CPU-final.log','OLD-ACTUAL-REVIEW.json']:guards[str(V/n)]=sha(V/n)
 # Fail locally on small artifact drift; leave archives and binary reads to the actual locked execution.
 verified=0
 for p,h in guards.items():
  if Path(p).stat().st_size<32*1024**2:
   if sha(p)!=h:raise ValueError('small input drift: '+p)
   verified+=1
 f=dict(sha256=guards,source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),original_frozen_API_unchanged=True,CPU_test_invocation='light CPU without Dlock, parent-authorized; no child model/GPU/compile',small_inputs_verified=verified,large_inputs_verified_here=False,large_inputs_required_under_execution_Dlock=True)
 (V/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n')
 ready=dict(status='READY_PARENT_OLD_MODEL_API_FRESH_UI_RETEST',writer_released=True,freeze_sha256=sha(V/'FREEZE.json'),sha256=guards,source_commit=f['source_commit'],CPP_binary_sha256='81c6a00b4e6aed90a712278623a4f11fb3f93846a5e22b6d133e493d83be389b',AtomicPASS=False,actual_UI_retest_pending=True,actual_thinking62_proof_required=True)
 (V/'READY.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps({k:ready[k] for k in ['status','writer_released','source_commit','freeze_sha256']}));print('READY SHA',sha(V/'READY.json'),'guards',len(guards))
if __name__=='__main__':main()

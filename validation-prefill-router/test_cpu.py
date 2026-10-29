from pathlib import Path
import re,subprocess,unittest
D=Path(__file__).resolve().parent;R=D.parent;s=(R/'src/dsv41/ds4_kernels.cu').read_text()
class Proof(unittest.TestCase):
 def test_inverse(self):
  old=subprocess.check_output(['git','show','1d868773:src/dsv41/ds4_kernels.cu'],cwd=R,text=True)
  self.assertEqual(re.sub(r'// PREFILL_ROUTER_REUSE(?:_DISPATCH)?_BEGIN\n.*?// PREFILL_ROUTER_REUSE(?:_DISPATCH)?_END\n\n?', '',s,flags=re.S),old)
 def test_fma_order(self):
  old=s[s.index('__global__ void __launch_bounds__(kThreads) gemv_f32_kernel('):s.index('// PREFILL_ROUTER_REUSE_BEGIN')]
  new=s[s.index('__global__ void __launch_bounds__(kThreads) gemv_f32_prefill_pair_kernel('):s.index('// PREFILL_ROUTER_REUSE_END')]
  seq=lambda body:re.findall(r'(\w+) = fmaf\(a\.([xyzw]), (\w+)\.\2, \1\);',body)
  self.assertEqual(seq(old),[('acc',x,'b')for x in 'xyzw'])
  self.assertEqual(seq(new),[('a0',x,'b0')for x in 'xyzw']+[('a1',x,'b1')for x in 'xyzw'])
  for body in [old,new]:self.assertIn('for (int c = lane; c < k / 4; c += 32)',body)
  self.assertIn('a0 = warp_sum(a0);',new);self.assertIn('if (second) a1 = warp_sum(a1);',new)
 def test_reach(self):
  for T in [1,16,17,31,4367,8192]:
   visits=[0]*T
   for block in range((T+1)//2):
    t=2*block;visits[t]+=1
    if t+1<T:visits[t+1]+=1
   self.assertEqual(visits,[1]*T)
 def test_guard(self):
  self.assertIn('T > 16 && T <= 8192 && n == 384 && k == 5120 && ldy == 1024',s)
  self.assertIn('std::strcmp(v, "1") == 0',s)
  self.assertIn('!defined(STRATA_USE_HIP) && !defined(STRATA_METAL)',s)
 def test_no_model_delta(self):
  self.assertEqual(subprocess.check_output(['git','diff','1d868773','--name-only','--','src/'],cwd=R,text=True).splitlines(),['src/dsv41/ds4_kernels.cu'])
 def test_fixture(self):
  fixture=(D/'main.inc').read_text()
  for term in ['{1,0},{16,1},{17,2},{31,0},{4367,0},{8192,0}','guard=%d old=%08x new=%08x','0x80000000','0x7fc00001','0xff800000','for(int mode:{0,1,1,0})']:
   self.assertIn(term,fixture)
if __name__=='__main__':unittest.main()

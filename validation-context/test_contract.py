import json,pathlib,sys,unittest
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools/atomic_context'))
from capacity import estimate,read_geometry,fingerprint,validate_completion
class Contracts(unittest.TestCase):
 def test_historical_geometry_formulas_only(self):
  g=read_geometry('/home/qwopqwop200/models/Abiray-DeepSeek-V4.1-IQ2_XXS/maya-ds4-q4/ds4.txt');p=estimate(g,1048576);self.assertEqual(p['source_cache_bytes'],964691392);self.assertEqual(p['source_caps'],{'2':524289,'8':524289,'14':524289,'20':1048577});self.assertEqual(p['index_slab_rows'],32)
  for context in (0,-1,1048577):
   with self.assertRaises(ValueError):estimate(g,context)
 def test_integer_fingerprint_contract(self):
  # Same explicit int32 littleendian FNV1a contract as C++ runtime. This is not a tokenized model fixture.
  h=fingerprint([0,1,128803,129279]);self.assertEqual(len(h),16);self.assertEqual(h,fingerprint([0,1,128803,129279]));self.assertNotEqual(h,fingerprint([0,1,128803,129278]))
 def test_synthetic_full_input_closure_and_negative_truncation(self):
  n=1000000;fixture=dict(ids=[0]*n,max_new=4096,context=1048576,expected={'begin':'TEST-ONLY'},input_fnv64='1234567890abcdef');q=dict(done=f'DONE 2 {n} 100 10 stop 0 0 0 0 0',tokens=[1,2],text='</think>{"begin":"TEST-ONLY"}')
  marker=f'ds4 context input: requested=1 input_tokens={n} fresh_from=0 prompt_read={n} engine_position={n} context=1048576 reuse_enabled=0 input_fnv64=1234567890abcdef full_input_position_match=1';progress=f'PP 8192 {n} 1 1 0\nPP {n} {n} 100 1 0\n'
  self.assertTrue(validate_completion(q,fixture,marker,progress)['pass_'])
  for bad in (marker.replace('engine_position=1000000','engine_position=131072'),marker.replace('fresh_from=0','fresh_from=128000'),marker.replace('reuse_enabled=0','reuse_enabled=1'),marker.replace('1234567890abcdef','0000000000000000')):self.assertFalse(validate_completion(q,fixture,bad,progress)['pass_'])
  self.assertFalse(validate_completion(q,fixture,marker,f'PP 131072 {n} 10 1 0\n')['pass_'])
  bad=dict(q,done=f'DONE 2 {n} 100 10 length 0 0 0 0 0');self.assertFalse(validate_completion(bad,fixture,marker,progress)['pass_'])
if __name__=='__main__':unittest.main(verbosity=2)

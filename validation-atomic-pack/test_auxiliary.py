from pathlib import Path
import importlib.util,tempfile,unittest,struct,json
R=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('pack',R/'tools/dsv41/pack_gguf.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class Auxiliary(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name);self.b=self.d/'bias.f32';self.b.write_bytes(struct.pack('<6f',*range(6)))
 def test_missing_only(self):
  files=['atomic.gguf'];old=('layers.0.ffn.gate.bias_vl','F32',0,17,12,[3]);ts=[old];r=p.add_vision_router_bias(files,ts,self.d,self.b,2,3);self.assertEqual(ts[0],old);self.assertEqual(ts[1],('layers.1.ffn.gate.bias_vl','F32',1,12,12,[3]));self.assertEqual(r['target_tensors_replaced'],0);self.assertEqual(self.b.read_bytes(),struct.pack('<6f',*range(6)))
 def test_all_existing_no_file_added(self):
  files=['atomic.gguf'];ts=[(f'layers.{i}.ffn.gate.bias_vl','F32',0,19,12,[3])for i in range(2)];r=p.add_vision_router_bias(files,ts,self.d,self.b,2,3);self.assertEqual(files,['atomic.gguf']);self.assertEqual(r['mapped_tensors'],[])
 def test_bad_size_type_shape(self):
  for ts in [[('layers.0.ffn.gate.bias_vl','BF16',0,0,6,[3])],[('layers.0.ffn.gate.bias_vl','F32',0,0,8,[2])]]:
   with self.assertRaises(SystemExit):p.add_vision_router_bias([],ts,self.d,self.b,2,3)
  with self.assertRaises(SystemExit):p.add_vision_router_bias([],[],self.d,self.b,3,3)
 def test_nonfinite_bounds(self):
  self.b.write_bytes(struct.pack('<6f',float('nan'),1,2,3,4,5))
  with self.assertRaises(SystemExit):p.add_vision_router_bias([],[],self.d,self.b,2,3)
  with self.assertRaises(SystemExit):p.add_vision_router_bias([],[],self.d,self.b,0,3)
 def test_tokenizer_declared_pre(self):
  kv={'tokenizer.ggml.tokens':['a','b','c'],'tokenizer.ggml.pre':'joyai-llm','tokenizer.ggml.bos_token_id':0,'tokenizer.ggml.eos_token_id':1,'tokenizer.ggml.padding_token_id':1}
  p.write_tokenizer(self.d/'tokenizer',kv,'template');x=json.loads((self.d/'tokenizer/tokenizer.json').read_text());self.assertEqual(x['pre'],'joyai-llm');self.assertEqual(x['special_ids'],{'bos':0,'eos':1,'pad':1});self.assertFalse(x['add_bos_token']);self.assertEqual(x['vocab_size'],3);self.assertEqual(x['gguf_special_ids'],{'tokenizer.ggml.bos_token_id':0,'tokenizer.ggml.eos_token_id':1,'tokenizer.ggml.padding_token_id':1});self.assertEqual((self.d/'tokenizer/chat_template.jinja').read_text(),'template')
 def test_new_ple_to_old_views_bytes(self):
  class G:pass
  g=G();g.path=self.d/'table';g.tensors=[('per_layer_token_embd.weight',20,[32,11],0)]
  p.split_ple(g,{'general.architecture':'deepseek41','deepseek41.engram.layers':[1,14],'deepseek41.engram.num_embeddings':[5,6]});self.assertEqual(g.tensors,[('blk.1.engram_embd.weight',20,[32,5],0),('blk.14.engram_embd.weight',20,[32,6],90)])
if __name__=='__main__':unittest.main()

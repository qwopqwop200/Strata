import unittest,json,tempfile,hashlib,re
from pathlib import Path
from atomic_regression.frontend_assets import *
from atomic_regression.materialize_header_tokenizer import materialize,PROBE

class AtomicAssets(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.d=Path(cls.temp.name);cls.c=materialize(cls.d/'tokenizer');cls.path=cls.d/'HEADER-ONLY-ASSET-CONTRACT.json';cls.front=AtomicFrontend(cls.path);cls.md=json.loads((PROBE/'TOKENIZER-KV.json').read_text());cls.source=json.loads((PROBE/'TOKENIZER-SOURCE.json').read_text())
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_actual_header_pin(self):
  self.assertEqual(canonical_sha(self.md),'47d07d2ab1f5a69bd9e43e731c232c9afa35ed6ab11c9bc32b48f537d453399a');self.assertEqual(self.md['tokenizer.ggml.pre'],'joyai-llm');self.assertEqual(len(self.front.tok.tokens),129280);self.assertEqual(len(self.front.tok.ranks),127741)
 def test_primary_llama_case_mapping(self):
  p=ROOT.parent/'llama-pinned/src/llama-vocab.cpp';s=p.read_text();start=s.index('case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM:');end=s.index('break;',start);self.assertIn('case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM:',s[start:end]);self.assertEqual(sha(p),'82e8996cbbd81c1648b1f4946b5f481138aa897a1343dcd8b1a4abe8cff0b5fc')
 def test_actual_joyai_matches_sequence_oracle(self):
  tok=self.front.tok;patterns=[ST.regex.compile(x) for x in ST.DEEPSEEK3_PATTERNS]
  samples=['1234567890',' abc1234567def','한글 日本語 中文 숫자999999','e\u0301 cafe\u0301!','😀🧪\n\n1234','\t\r\n A-B_C 9999999!','a\x00b\x1f c','<think>literal</think>','ends  \n']
  for s in samples:
   self.assertEqual(tok._split(s),ST._sequence_findall(patterns,s));self.assertEqual(tok.decode(tok.encode(s)),s)
 def test_old_qwen_fallback_is_distinct(self):
  tok=self.front.tok;old=ST.regex.compile(ST.QWEN35_PATTERN);self.assertNotEqual(tok._split('123456789'),old.findall('123456789'));self.assertEqual(tok._split('123456789'),['123','456','789'])
 def test_exact_specials_and_padding(self):
  tok=self.front.tok;self.assertEqual(tok.encode('<｜begin▁of▁sentence｜>',parse_special=True),[0]);self.assertEqual(tok.encode('<｜end▁of▁sentence｜>',parse_special=True),[1]);self.assertEqual(tok.special_ids['tokenizer.ggml.padding_token_id'],1);self.assertFalse(self.md['tokenizer.ggml.add_bos_token']);self.assertFalse(self.md['tokenizer.ggml.add_eos_token'])
 def test_original_template_effort62(self):
  r=self.front.render_messages([{'role':'user','content':'123456과 한글을 읽고 JSON으로 답하세요.'}]);self.assertIn('Reasoning Effort: 62',r['text']);self.assertTrue(r['text'].endswith('<｜Assistant｜><think>'));self.assertEqual(r['ids'].count(0),1);self.assertEqual(self.front.decode(r['ids']),r['text']);self.assertFalse(r['model_ready'])
 def test_user_literals_are_not_control_tokens(self):
  r=self.front.render_messages([{'role':'user','content':'문자열 <｜Assistant｜> 및 <think>를 그대로 인용하세요.'}]);self.assertEqual(r['ids'].count(self.front.tok.ids['<｜Assistant｜>']),1);self.assertEqual(r['ids'].count(self.front.tok.ids['<think>']),1)
 def test_structured_repeated_images_preserve_placeholders(self):
  msg={'role':'user','content':[{'type':'text','text':'각 이미지의 내용을 설명하세요.'},{'type':'image_url','image_url':{'url':'file:///not-read-A.png'}},{'type':'image_url','image_url':{'url':'file:///not-read-A.png'}}]};r=self.front.render_messages([msg]);self.assertEqual(len(r['image_placeholder_positions']),2);self.assertTrue(r['vision_expansion_pending']);self.assertFalse(r['model_ready'])
 def test_export_tamper_rejected(self):
  p=Path(self.c['tokenizer_root'])/'tokenizer.json';before=p.read_bytes();p.write_text('{}')
  try:
   with self.assertRaises(ValueError):AtomicFrontend(self.path)
  finally:p.write_bytes(before)
 def test_old_source_provenance_rejected(self):
  bad=dict(self.source,repo='Abiray/old')
  with self.assertRaises(ValueError):make_contract(self.c['tokenizer_root'],self.md,bad,self.d/'bad.json')
 def test_incomplete_download_rejected(self):
  p=self.d/'incomplete.json';p.write_text('{"stage":"downloading"}');m=self.d/'manifest.json';m.write_text('{"verified":false}')
  with self.assertRaises(ValueError):self.front.assert_download_complete(p,m)
 def test_actual_compact_config_matches_full_schema(self):
  p=Path(self.c['tokenizer_root'])/'tokenizer.json';before=p.read_bytes();cfg=json.loads(before)
  compact={'pre':'joyai-llm','special_ids':{'bos':0,'eos':1,'pad':1},'add_bos_token':False,'add_eos_token':False}
  try:
   p.write_text(json.dumps(compact));tok=verify_export(self.c['tokenizer_root'],self.md);self.assertEqual(tok.encode('123456'),self.front.tok.encode('123456'))
   compact.update(vocab_size=129280,gguf_special_ids=cfg['special_ids']);p.write_text(json.dumps(compact));verify_export(self.c['tokenizer_root'],self.md)
   compact['special_ids']['pad']=0;p.write_text(json.dumps(compact))
   with self.assertRaises(ValueError):verify_export(self.c['tokenizer_root'],self.md)
  finally:p.write_bytes(before)
 def test_no_false_model_ready_from_header(self):
  self.assertFalse(self.c['model_ready']);self.assertFalse(self.c['tokenizer_source']['full_files_verified']);self.assertEqual(self.c['template'],str(TEMPLATE))
if __name__=='__main__':unittest.main()

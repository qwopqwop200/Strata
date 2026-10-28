"""Actual original ISTA template and Atomic tokenizer through API/Service, CPU mocks only."""
import json,tempfile,unittest,urllib.request,urllib.error
from pathlib import Path
from serve import frontend
from serve.frontend import ChatTemplate
from serve.server import Service,serve,clean_shared_defaults
from serve.test_dsv41 import DsEngine
from atomic_regression.materialize_header_tokenizer import materialize
from atomic_regression.frontend_assets import AtomicFrontend,TEMPLATE
class NumericEffort(unittest.TestCase):
 def tearDown(self):frontend.use_family(None)
 def test_deepseek_numeric_string(self):
  frontend.use_family('deepseek41')
  for value in ['1','62','100']:self.assertEqual(frontend.effort_kwargs(value),{'reasoning_effort':int(value)})
 def test_reject_noncanonical_numeric(self):
  frontend.use_family('deepseek41')
  for value in ['0','101','062','+62','-1','62.0',' 62','62 ','٦٢']:
   with self.assertRaises(ValueError):frontend.effort_kwargs(value)
 def test_other_families_unchanged(self):
  frontend.use_family(None);self.assertEqual(frontend.effort_kwargs('medium'),{'reasoning_effort':'medium'})
  for value in ['62',62]:
   with self.assertRaises(ValueError):frontend.effort_kwargs(value)
 def test_existing_medium_keeps62_with_original_template(self):
  frontend.use_family('deepseek41');self.assertEqual(frontend.effort_kwargs('medium'),{'reasoning_effort':62});self.assertEqual(frontend.budget_effort(4096),{'reasoning_effort':62})
 def test_false_and_boundaries_unchanged(self):
  frontend.use_family('deepseek41');self.assertEqual(frontend.effort_kwargs(False),{'enable_thinking':False});self.assertEqual(frontend.effort_kwargs(62),{'reasoning_effort':62})
  for value in [0,101,True,62.0]:
   with self.assertRaises(ValueError):frontend.effort_kwargs(value)
 def test_deepseek_config_and_shared_numeric(self):
  for x in [62,"62",1,100]:self.assertEqual(clean_shared_defaults({'reasoning_effort':x},family='deepseek41'),{'reasoning_effort':int(x)})
  for x in [62,"62"]:
   with self.assertRaises(ValueError):clean_shared_defaults({'reasoning_effort':x})
  for x in [0,101,"062","+62",True]:
   with self.assertRaises(ValueError):clean_shared_defaults({'reasoning_effort':x},family='deepseek41')
class ActualTemplateService(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();p=Path(cls.temp.name);materialize(p/'tokenizer');cls.front=AtomicFrontend(p/'HEADER-ONLY-ASSET-CONTRACT.json');cls.tok=cls.front.tok;cls.engine=DsEngine(cls.tok,'검사.</think>{"ok":true}');cls.svc=Service(cls.engine,cls.tok,ChatTemplate(TEMPLATE));cls.svc.set_family('deepseek41');cls.http=serve(cls.svc,port=0);cls.url='http://127.0.0.1:'+str(cls.http.server_address[1])
 @classmethod
 def tearDownClass(cls):cls.http.shutdown();cls.http.server_close();frontend.use_family(None);cls.temp.cleanup()
 def setUp(self):frontend.use_family('deepseek41')
 def post(self,body):
  req=urllib.request.Request(self.url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=10) as r:return r.status,json.loads(r.read())
  except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
 def test_original_template_both_thinking_options_identical(self):
  messages=[{'role':'user','content':'숫자123456을 읽고 답하세요.'}];t=self.svc.template;a=t.render(messages,reasoning_effort=62,enable_thinking=True);b=t.render(messages,reasoning_effort=62,thinking_mode='thinking');self.assertEqual(a,b);self.assertIn('Reasoning Effort: 62 ',a)
 def test_service_and_direct_frontend_exact_IDS(self):
  messages=[{'role':'user','content':'한글 123456 및 <think>문자열을 읽으세요.'}];m,t,k=frontend.openai_to_messages({'messages':messages,'reasoning_effort':'62'});ids=self.svc.encode_prompt(m,t,k,store=False);direct=self.front.render_messages(messages);self.assertEqual(ids,direct['ids'])
 def test_real_HTTP_numericstring62(self):
  body={'model':'AtomicChat-DeepSeek-V4.1-IQ2_XXS','messages':[{'role':'user','content':'안녕하세요.'}],'reasoning_effort':'62','temperature':0,'max_tokens':64};status,r=self.post(body);self.assertEqual(status,200,r);self.assertIn('Reasoning Effort: 62 ',self.tok.decode(self.engine.last_prompt));self.assertEqual(r['choices'][0]['finish_reason'],'stop')
 def test_real_HTTP_template_kwargs_enable_thinking(self):
  body={'messages':[{'role':'user','content':'답하세요.'}],'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'62'},'max_tokens':64};status,r=self.post(body);self.assertEqual(status,200,r);self.assertIn('Reasoning Effort: 62 ',self.tok.decode(self.engine.last_prompt));self.assertTrue(self.tok.decode(self.engine.last_prompt).endswith('<｜Assistant｜><think>'))
 def test_real_HTTP_unasked_effort_uses_config62(self):
  prior=self.svc.config_effort
  try:
   self.svc.config_effort=clean_shared_defaults({'reasoning_effort':62},family=self.svc.family)['reasoning_effort']
   status,r=self.post({'messages':[{'role':'user','content':'기본 노력값으로 답하세요.'}],'max_tokens':64});self.assertEqual(status,200,r);self.assertIn('Reasoning Effort: 62 ',self.tok.decode(self.engine.last_prompt))
  finally:self.svc.config_effort=prior
 def test_real_HTTP_invalid_effort400(self):
  status,r=self.post({'messages':[{'role':'user','content':'답하세요.'}],'reasoning_effort':'101','max_tokens':64});self.assertEqual(status,400,r)
if __name__=='__main__':unittest.main()

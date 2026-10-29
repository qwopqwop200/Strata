import json,subprocess,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class UI(unittest.TestCase):
 def test_actual_ui_default_and_explicit_choices(self):
  source=(R/'serve/web/app.js').read_text();start=source.index('const DEFAULTS =');end=source.index('if (samplingUntouched(settings))',start);body=source[start:end]
  js='''const assert=require('node:assert/strict'); const children=['none','low','medium','high'].map(v=>({dataset:{v},setAttribute(){}})); const $=()=>({children,appendChild(b){children.push(b)}});const document={createElement:()=>({dataset:{},setAttribute(){}})};const outputs=()=>{};const store={get:()=>(SAVED)};let healthDefaultsLoaded=false;'''
  for saved,default,expected in [({},62,'62'),({},None,'high'),({'thinking':'low'},62,'low'),({'thinking':'high'},62,'high'),({},'high','high'),({},101,'high')]:
   code=js.replace('SAVED',json.dumps(saved))+body+('assert.deepEqual(thinkingRequestFields(),{});' if not saved else '')+f'healthDefaultsLoaded=true;applyThinkingDefault({json.dumps(default)});assert.equal(settings.thinking,{json.dumps(expected)});assert.equal(thinkingRequestFields().reasoning_effort,thinkingRequestValue(settings.thinking));assert.equal(thinkingRequestValue("62"),62);assert.equal(thinkingRequestValue("high"),"high");'
   subprocess.run(['node','-e',code],check=True,capture_output=True,timeout=5)
  self.assertIn('stream: true, ...thinkingRequestFields()',source)
  self.assertIn('thinking: segValue("s-thinking") || settings.thinking',source)
 def test_server_default_shared_precedence(self):
  from serve.server import Service
  class Fake:shared={};config_effort=62
  self.assertEqual(Service.effort_default(Fake()),62)
  Fake.shared={'reasoning_effort':'low'};self.assertEqual(Service.effort_default(Fake()),'low')
if __name__=='__main__':unittest.main()

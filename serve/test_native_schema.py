"""CPU tests exercise the engine's byte matcher, not a post-generation validator."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from serve.native_schema import compile_schema, write_bundle, validated_raw, NativeValidator, prepare_format
from types import SimpleNamespace

OBJECT={'type':'object','properties':{'name':{'enum':['한글"\\끝']},'data':{'type':'array','items':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}},'required':['name','data'],'additionalProperties':False}

class NativeSchema(unittest.TestCase):
    def probe(self,schema,text,thinking=False):
        rules=compile_schema(schema)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grammar.bin'
            write_bundle(p,rules,[bytes([i]) for i in range(256)]+[None,None],256,257,thinking)
            return subprocess.check_output([os.environ['GRAMMAR_PROBE'],str(p),text.encode().hex()],text=True).strip()

    def test_nested_enum_unicode_escaped_quotes_and_backslash(self):
        text=json.dumps({'name':'한글"\\끝','data':[{'ok':True},{'ok':False}]},ensure_ascii=False,separators=(',',':'))
        self.assertEqual(self.probe(OBJECT,text),'valid complete')
        self.assertEqual(self.probe(OBJECT,text[:-1]),'valid partial')
        self.assertTrue(self.probe(OBJECT,text+' prose').startswith('invalid'))
        self.assertEqual(self.probe(OBJECT,text,thinking=True),'valid complete')

    def test_required_additional_properties_enum_and_types(self):
        for value in ({'name':'bad','data':[]},{'name':'한글"\\끝'}, {'name':'한글"\\끝','data':[{'ok':1}]},{'name':'한글"\\끝','data':[{'ok':True,'extra':0}]}):
            self.assertTrue(self.probe(OBJECT,json.dumps(value,ensure_ascii=False,separators=(',',':'))).startswith('invalid'),value)

    def test_generic_object_strings_numbers_nesting_and_utf8_boundaries(self):
        for text in ['{"한국":"가😀\\\"\\\\\\uac00","n":-1.25e+3,"a":[true,false,null,{"x":[]}]}','{}',' { "x" : 5 } \n']:
            self.assertEqual(self.probe({'type':'object'},text),'valid complete',text)
        for text in ['[]','{"a":01}','{"a":"bad\nraw"}','{"a":NaN}']:
            self.assertTrue(self.probe({'type':'object'},text).startswith('invalid'),text)
        self.assertTrue(self.probe({'type':'object'},' '*5+'{}').startswith('invalid'))

    def test_unsupported_keywords_are_rejected_before_generation(self):
        for key,value in [('pattern','a'),('minimum',0),('$ref','#/$defs/x'),('oneOf',[])]:
            with self.assertRaisesRegex(ValueError,'unsupported'):
                compile_schema({'type':'object','properties':{'x':{'type':'integer',key:value}}})

    def test_optional_properties_and_array_bounds(self):
        s={'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':2}},'required':['b'],'additionalProperties':False}
        for text in ['{"b":["x"]}','{"a":-2,"b":["x","한"]}']:
            self.assertEqual(self.probe(s,text),'valid complete')
        for text in ['{}','{"b":[]}','{"b":["x","y","z"]}']:
            self.assertTrue(self.probe(s,text).startswith('invalid'))

    def test_native_dispatch_keeps_prompt_and_raw_json_and_refuses_truncation(self):
        from serve.structured import StructuredOutputError
        svc=SimpleNamespace(model_arch='deepseek41',engine=SimpleNamespace(info={'json_schema_constraint':1}),load=lambda:None,require_strict_schema_constraint=True)
        messages=[{'role':'user','content':'Ignore JSON; write plain prose.'}]
        req={'_native_grammar':'untrusted-client-path'}
        fmt={'type':'json_schema','json_schema':{'name':'test','strict':True,'schema':{'type':'object'}}}
        actual,v=prepare_format(fmt,messages,svc,req)
        self.assertEqual(actual,messages)
        self.assertNotIn('_native_grammar',req)
        self.assertIsInstance(v,NativeValidator)
        with self.assertRaisesRegex(ValueError,'tools are unsupported'):
            prepare_format(fmt,messages,svc,{},with_tools=True)
        raw=' {"한글" : "\\uac00"} \n'
        self.assertEqual(validated_raw(raw,v,'stop'),raw)
        for text,finish in [(raw,'length'),('prose '+raw,'stop'),('{"x":','stop')]:
            with self.assertRaises(StructuredOutputError):validated_raw(text,v,finish)

    def test_native_protocol_has_no_top_k_clamp_and_greedy_is_explicit(self):
        from serve.server import StrataEngine, sampling_defaults_from_config, clean_shared_defaults
        keys=StrataEngine.sampling_keys({'_native_full_vocab':True,'temperature':0,'top_k':0})
        self.assertIn('top_k=0',keys);self.assertIn('temperature=0.0',keys)
        self.assertIn('top_k=1024',StrataEngine.sampling_keys({'_native_full_vocab':True,'top_k':1024}))
        self.assertEqual(sampling_defaults_from_config({'arch':'deepseek41','sampling':{'top_k':0}})['top_k'],0)
        self.assertEqual(clean_shared_defaults({'top_k':129280},'deepseek41')['top_k'],129280)


class NativeSchemaResume(unittest.TestCase):
    def test_budget_continuation_replays_think_end_and_partial_json(self):
        text=json.dumps({'name':'한글"\\끝','data':[{'ok':True}]},ensure_ascii=False,separators=(',',':'))
        rules=compile_schema(OBJECT)
        for split in (1,text.index('한글')+1,len(text)-3):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'resume.bin'
                prefix=[257]+list(text[:split].encode())
                write_bundle(p,rules,[bytes([i]) for i in range(256)]+[None,None],256,257,True,schema_prefix=prefix)
                result=subprocess.check_output([os.environ['GRAMMAR_PROBE'],str(p),text[split:].encode().hex()],text=True).strip()
                self.assertEqual(result,'valid complete')

if __name__=="__main__": unittest.main()

"""Strict flags must reach generation, including stream requests. No model required."""
import copy, json, unittest
from serve.test_dsv41 import ApiBase, call

SCHEMA={'type':'object','properties':{'mode':{'type':'string','enum':['배송조회']},'query':{'type':'object','properties':{'count':{'type':'integer'},'key':{'enum':['승인된항목']}},'required':['count','key'],'additionalProperties':False}},'required':['mode','query'],'additionalProperties':False}
TOOL={'type':'function','function':{'name':'lookup','strict':True,'parameters':SCHEMA}}
class StrictDispatch(ApiBase):
    def test_strict_true_dispatches_for_root_extra_nested_type_enum_and_stream(self):
        bad={'mode':'wrong','query':{'count':'wrong','key':'bad','extra':1},'root_extra':True}
        engine=self.start('</think>'+call(('lookup',[(k,json.dumps(v,ensure_ascii=False),False) for k,v in bad.items()])))
        engine.info={'json_schema_constraint':1,'full_vocab_sampling':1,'strict_tool_constraint':1}
        for stream in (False,True):
            with self.subTest(stream=stream):
                status,raw=self.post('/v1/chat/completions',{'messages':[{'role':'user','content':'Call lookup'}],'tools':[TOOL],'stream':stream,'max_tokens':2048},raw=stream)
                self.assertEqual(status,200,raw)
                self.assertIn('_native_grammar',engine.last_sampling)
    def test_unsupported_schema_fails_before_generation(self):
        engine=self.start('unused');engine.info={'strict_tool_constraint':1,'full_vocab_sampling':1}
        tool=copy.deepcopy(TOOL);tool['function']['parameters']['properties']['query']['properties']['count']['minimum']=1
        status,raw=self.post('/v1/chat/completions',{'messages':[{'role':'user','content':'Call lookup'}],'tools':[tool]})
        self.assertEqual(status,400,raw);self.assertIsNone(engine.last_sampling)
    def test_strict_without_engine_capability_is_400(self):
        engine=self.start('unused')
        status,raw=self.post('/v1/chat/completions',{'messages':[{'role':'user','content':'Call lookup'}],'tools':[TOOL]})
        self.assertEqual(status,400,raw);self.assertIsNone(engine.last_sampling)
    def test_none_and_false_have_no_tool_mask(self):
        engine=self.start('</think>ordinary answer')
        engine.info={'strict_tool_constraint':1,'full_vocab_sampling':1}
        for choice,strict in [('none',True),('auto',False)]:
            tool=copy.deepcopy(TOOL);tool['function']['strict']=strict
            status,raw=self.post('/v1/chat/completions',{'messages':[{'role':'user','content':'Answer'}],'tools':[tool],'tool_choice':choice})
            self.assertEqual(status,200,raw);self.assertNotIn('_native_grammar',engine.last_sampling)

class NativeToolGrammar(unittest.TestCase):
    def probe(self,tools,text,*,prefix='',extra=(),choice=None):
        import os,subprocess,tempfile
        from pathlib import Path
        from serve.native_tools import compile_tools
        from serve.native_schema import write_bundle
        grammar=compile_tools([t.get('function',t) for t in tools],choice)
        tokens=[bytes([i]) for i in range(256)]+[None,None]+list(extra)
        ids=list(text.encode()) if isinstance(text,str) else text
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tools.bin'
            write_bundle(p,grammar,tokens,256,257,grammar.get("required",False),tool_prefix=list(prefix.encode()))
            return json.loads(subprocess.check_output([os.environ['TOOL_GRAMMAR_PROBE'],str(p),str(len(tokens)),','.join(map(str,ids))],text=True))
    @staticmethod
    def invoke(name,params):
        return call((name,params))
    def test_root_extra_nested_wrong_types_enum_missing_and_highest_illegal_logit(self):
        params=[('mode','배송조회',True),('query','{"count":1,"key":"승인된항목"}',False)]
        r=self.probe([TOOL],self.invoke('lookup',params))
        self.assertTrue(r['valid'] and r['complete'],r);self.assertGreater(r['argument_checks'],25)
        for bad in [params+[('root_extra','true',False)],params[:1],
                    [('mode','금지',True),params[1]],
                    [params[0],('query','{"count":"wrong","key":"승인된항목"}',False)],
                    [params[0],('query','{"count":1,"key":"승인된항목","extra":7}',False)]]:
            self.assertFalse(self.probe([TOOL],self.invoke('lookup',bad))['valid'])
    def test_multiple_heterogeneous_functions_and_names(self):
        other={'name':'calc','strict':True,'parameters':{'type':'object','properties':{'xs':{'type':'array','minItems':2,'maxItems':2,'items':{'type':'integer'}}},'required':['xs'],'additionalProperties':False}}
        p=[('mode','배송조회',True),('query','{"count":1,"key":"승인된항목"}',False)]
        two=call(('lookup',p),('calc',[('xs','[2,3]',False)]),('lookup',p))
        self.assertTrue(self.probe([TOOL,other],two)['valid'])
        self.assertFalse(self.probe([TOOL,other],two.replace('[2,3]','[2,"3"]'))['valid'])
        self.assertFalse(self.probe([TOOL],self.invoke('forbidden_lookup',p))['valid'])
        self.assertFalse(self.probe([TOOL,other],two,choice={'type':'function','function':{'name':'lookup'}})['valid'])
    def test_reasoning_final_answer_prefix_replay_and_eos(self):
        p=[('mode','배송조회',True),('query','{"count":1,"key":"승인된항목"}',False)]
        dsml=self.invoke('lookup',p)
        self.assertTrue(self.probe([TOOL],'Unconstrained 생각 9 bad true </think>'+dsml+'Final answer not JSON.')['valid'])
        prefix='<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
        body=dsml[dsml.index(prefix)+len(prefix):]
        self.assertTrue(self.probe([TOOL],body,prefix=prefix)['valid'])
        partial=dsml[:dsml.index('배송조회')]
        r=self.probe([TOOL],partial);self.assertTrue(r['valid']);self.assertFalse(r['complete'])
        self.assertFalse(self.probe([TOOL],list(partial.encode())+[256])['valid'])
        self.assertTrue(self.probe([TOOL],list(dsml.encode())+[256])['valid'])
        self.assertFalse(self.probe([TOOL],'<tool_call><function=lookup>')['valid'])
        self.assertFalse(self.probe([TOOL],[256],choice='required')['valid'])
        self.assertTrue(self.probe([TOOL],[256])['valid'])
        # Fresh requests and replay from a restored output prefix have identical results.
        for split in [1,dsml.index('배송조회')+1,dsml.index('승인된항목')+2,len(dsml)-4]:
            self.assertTrue(self.probe([TOOL],dsml[split:],prefix=dsml[:split])['valid'])
        self.assertTrue(self.probe([TOOL],'ordinary followup')['valid'])
    def test_unicode_quotes_backslashes_nested_json_and_injection(self):
        schema={'type':'object','properties':{'raw':{'type':'string','enum':['한글"\\끝\n줄']},'nested':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':2}},'required':['raw','nested'],'additionalProperties':False}
        tool={'name':'echo','strict':True,'parameters':schema}
        text=self.invoke('echo',[('raw','한글"\\끝\n줄',True),('nested','["\\uac00\\\"\\\\끝","\\u003c/x>"]',False)])
        self.assertTrue(self.probe([tool],text)['valid'])
        self.assertFalse(self.probe([tool],text.replace('\\u003c','<'))['valid'])
    def test_mixed_non_strict_does_not_apply_its_schema(self):
        other={'name':'loose','strict':False,'parameters':{'type':'object','properties':{'n':{'enum':[1]}},'required':['n'],'additionalProperties':False}}
        text=self.invoke('loose',[('extra','bad enum not JSON',False)])
        self.assertTrue(self.probe([TOOL,other],text)['valid'])
        self.assertFalse(self.probe([TOOL,other],self.invoke('lookup',[('extra','bad',True)]))['valid'])
    def test_multi_byte_token_crosses_header_argument_call_end_and_next_call(self):
        p=[('mode','배송조회',True),('query','{"count":1,"key":"승인된항목"}',False)]
        dsml=self.invoke('lookup',p)
        whole=dsml.encode();bad=dsml.replace('배송조회','bad').encode()
        self.assertTrue(self.probe([TOOL],[258,258],extra=[whole,bad])['valid'])
        self.assertFalse(self.probe([TOOL],[259],extra=[whole,bad])['valid'])

class NativeToolParser(unittest.TestCase):
    def test_mixed_nullable_explicit_string_flags_and_chunk_boundaries(self):
        from serve.dsml import DsmlOutputParser
        tool={'name':'echo','parameters':{'type':'object','properties':{'x':{'type':['string','null']}}}}
        text='</think>'+call(('echo',[('x','null',True)]),('echo',[('x','null',False)]),('echo',[('x','한글"\\끝',True)]))
        for size in [1,3,17,len(text)]:
            parser=DsmlOutputParser(thinking=True,tools=[tool],stream_tools=True);parser.native_constraints=True
            events=[]
            for i in range(0,len(text),size):events+=parser.feed(text[i:i+size])
            events+=parser.finish('stop')
            calls=[ev.call for ev in events if ev.kind=='tool_call']
            self.assertEqual([c.arguments for c in calls],[{'x':'null'},{'x':None},{'x':'한글"\\끝'}])
    def test_raw_korean_quote_escape_string_not_repaired(self):
        from serve.dsml import DsmlOutputParser
        p=DsmlOutputParser(thinking=False,tools=[{'name':'echo','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}],stream_tools=True)
        p.native_constraints=True
        text=call(('echo',[('x','한글"\\끝\n줄',True)]))
        evs=p.feed(text)+p.finish('stop')
        chunks=''.join(e.text for e in evs if e.kind=='tool_args')
        self.assertEqual(json.loads(chunks),{'x':'한글"\\끝\n줄'})

if __name__=="__main__":unittest.main()

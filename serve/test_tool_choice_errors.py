"""Reject choices the route cannot honor before sending SSE or generating tokens."""
import http.client
import json
from pathlib import Path
import unittest
from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve

TOOLS=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{}}}}]

class ToolChoiceErrors(unittest.TestCase):
    def setUp(self):
        tok=ByteTokenizer(); self.engine=MockEngine(tok,'unexpected generation')
        self.httpd=serve(Service(self.engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja'))),port=0)

    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close()

    def post(self,path,body):
        c=http.client.HTTPConnection('127.0.0.1',self.httpd.server_address[1],timeout=10)
        try:
            c.request('POST',path,json.dumps(body),{'Content-Type':'application/json'})
            r=c.getresponse(); return r.status,r.getheader('Content-Type'),json.loads(r.read())
        finally: c.close()

    def test_chat_rejects_invalid_or_impossible_choices_before_stream(self):
        for stream in (False,True):
            for choice,tools in [('sometimes',TOOLS),({'type':'banana'},TOOLS),({'type':'function'},TOOLS),
                                 ({'type':'function','function':{'name':'missing'}},TOOLS),('required',[]),(False,TOOLS)]:
                with self.subTest(stream=stream,choice=choice):
                    status,ctype,body=self.post('/v1/chat/completions',{'messages':[{'role':'user','content':'hi'}],
                        'tools':tools,'tool_choice':choice,'stream':stream,'max_tokens':64})
                    self.assertEqual(status,400,body); self.assertIn('application/json',ctype)
                    self.assertIn('tool_choice',body['error']['message'])
                    self.assertFalse(getattr(self.engine,'last_prompt',None))

    def test_responses_rejects_unimplemented_choices_before_stream(self):
        for stream in (False,True):
            for choice in ('sometimes','required',{'type':'function','name':'lookup'},{'type':'allowed_tools','tools':[]}):
                with self.subTest(stream=stream,choice=choice):
                    status,ctype,body=self.post('/v1/responses',{'input':'hi','tools':[{'type':'function',**TOOLS[0]['function']}],
                        'tool_choice':choice,'stream':stream,'max_output_tokens':64})
                    self.assertEqual(status,400,body); self.assertIn('application/json',ctype)
                    self.assertEqual(body['error']['param'],'tool_choice')
                    self.assertFalse(getattr(self.engine,'last_prompt',None))

if __name__=='__main__': unittest.main()

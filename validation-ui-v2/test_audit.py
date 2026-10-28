import hashlib,io,json,os,struct,tempfile,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class Audit(unittest.TestCase):
 def test_actual_direct_writes_split_stop_quit_and_token_decode(self):
  source=(R/'tools/atomic_vision/server_entry.py').read_text();core=R.parent/'atomic-agent2-vision-v1/tools/atomic_vision'
  import sys;sys.path.insert(0,str(core));from tokenizer_assets import load
  tok=load(R.parent/'atomic-agent2-vision-v1/validation-atomic-vision/old-smoke-1m-v3/tokenizer')
  rows=[];ns=dict(hashlib=hashlib,json=json,tokenizer=tok,re=__import__('re'),note=lambda kind,**kw:rows.append(dict(event=kind,**kw)),packet_meta=lambda line:dict(test_packet=True))
  body=source[source.index('class AuditedInput:'):source.index('original_done=')];exec(body,ns)
  ids=tok.encode('Reasoning Effort: 62 (range 1-100, the higher the value, the more thorough the reasoning)\n\n',parse_special=True)
  raw=io.StringIO();proxy=ns['AuditedInput'](raw,123)
  line='GENI 1024 /tmp/fixture.dsi '+','.join(map(str,ids))+'\n';proxy.write(line[:17]);proxy.write(line[17:]);proxy.flush();proxy.write('STOP\nQUIT\n')
  self.assertEqual(raw.getvalue(),line+'STOP\nQUIT\n');self.assertEqual([r['command']for r in rows],['GENI','STOP','QUIT']);self.assertEqual(rows[0]['input_ids'],ids);self.assertEqual(rows[0]['reasoning_effort_observed'],[62]);self.assertTrue(rows[0]['write_returned'])
  rows.clear();ids=tok.encode('Reasoning Effort: 75 (range 1-100,',parse_special=True);proxy.write('GEN 10 '+','.join(map(str,ids))+'\n');self.assertEqual(rows[0]['reasoning_effort_observed'],[75])
if __name__=='__main__':unittest.main()

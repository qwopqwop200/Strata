"""Generate a bound text prompt from the real exported Atomic tokenizer/template."""
import argparse,json
from pathlib import Path
from frontend_assets import AtomicFrontend
p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--messages',required=True);p.add_argument('--out',required=True);a=p.parse_args()
f=AtomicFrontend(a.contract);result=f.render_messages(json.loads(Path(a.messages).read_text()));out=Path(a.out)
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['text','ids']}))

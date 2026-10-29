"""JSON Schema subset -> byte CFG consumed by the native DeepSeek sampler.

No output is repaired. Object properties are generated in schema declaration order.
Unsupported semantic keywords fail before generation instead of being silently ignored.
"""
import json
import struct
import hashlib
import tempfile
from pathlib import Path

ANNOTATIONS={'title','description','$schema','examples','default'}
KEYS=ANNOTATIONS|{'type','properties','required','additionalProperties','items','minItems','maxItems','enum','const'}

class Compiler:
    def __init__(self):
        self.rules=[]
        # Bound formatting whitespace so a conflicting prompt cannot avoid the required JSON forever.
        white=self.new([[self.rng(9,10)],[13],[32]])
        self.ws=self.new([[]]+[[white]*n for n in range(1,5)])
        self.hex=self.new([[self.rng(48,57)],[self.rng(65,70)],[self.rng(97,102)]])
        self.digits=self.new();self.set(self.digits,[[],[self.rng(48,57),self.digits]])
        self.integer=self.new([[48],[self.rng(49,57),self.digits]])
        sign=self.new([[],[45]])
        fraction=self.new([[],[46,self.rng(48,57),self.digits]])
        esign=self.new([[],[43],[45]])
        exponent=self.new([[],[self.rng(69,69),esign,self.rng(48,57),self.digits],[101,esign,self.rng(48,57),self.digits]])
        self.int_value=self.new([[sign,self.integer,self.ws]])
        self.number=self.new([[sign,self.integer,fraction,exponent,self.ws]])
        tail=self.rng(128,191)
        esc=self.new([[x] for x in b'"\\/bfnrt']+[[117,self.hex,self.hex,self.hex,self.hex]])
        char=self.new([[self.rng(32,33)],[self.rng(35,91)],[self.rng(93,127)],[92,esc],
                       [self.rng(194,223),tail],[224,self.rng(160,191),tail],
                       [self.rng(225,236),tail,tail],[237,self.rng(128,159),tail],
                       [self.rng(238,239),tail,tail],[240,self.rng(144,191),tail,tail],
                       [self.rng(241,243),tail,tail,tail],[244,self.rng(128,143),tail,tail]])
        body=self.new();self.set(body,[[],[char,body]])
        self.string=self.new([[34,body,34,self.ws]])
        self.boolean=self.new([self.literal(True),self.literal(False)])
        self.null=self.new([self.literal(None)])
        self.any=self.new();self.object=self.new();self.array=self.new()
        members=self.new();more=self.new([[],[44,self.ws,members]])
        self.set(members,[[self.string,58,self.ws,self.any,more]])
        optional=self.new([[],[members]])
        self.set(self.object,[[123,self.ws,optional,125,self.ws]])
        elements=self.new();rest=self.new([[],[44,self.ws,elements]])
        self.set(elements,[[self.any,rest]])
        optional=self.new([[],[elements]])
        self.set(self.array,[[91,self.ws,optional,93,self.ws]])
        self.set(self.any,[[r] for r in [self.object,self.array,self.string,self.number,self.boolean,self.null]])

    @staticmethod
    def rng(lo,hi):return -1-(lo*256+hi)
    def new(self,alts=None):
        r=256+len(self.rules);self.rules.append(alts or []);return r
    def set(self,r,alts):self.rules[r-256]=alts
    def literal(self,value):
        return list(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())+[self.ws]

    def value(self,s,depth=0):
        if depth>64:raise ValueError('unsupported schema nesting beyond 64')
        if not isinstance(s,dict):raise ValueError('unsupported non-object schema')
        bad=set(s)-KEYS
        if bad:raise ValueError('unsupported schema keyword: '+sorted(bad)[0])
        typ=s.get('type')
        if 'enum' in s or 'const' in s:
            values=s.get('enum',[s.get('const')])
            if not isinstance(values,list) or not values:raise ValueError('enum must be a nonempty array')
            from jsonschema import Draft202012Validator
            validator=Draft202012Validator(s)
            if not all(validator.is_valid(v) for v in values):raise ValueError('enum values conflict with schema')
            # Validate every nested keyword even when the enum determines the whole serialized value.
            self.value({k:v for k,v in s.items() if k not in ('enum','const')},depth+1)
            return self.new([self.literal(v) for v in values])
        if isinstance(typ,list):
            return self.new([[self.value({**s,'type':t},depth+1)] for t in typ])
        relevant={'object':{'properties','required','additionalProperties'},'array':{'items','minItems','maxItems'}}
        unexpected=set(s)-ANNOTATIONS-{'type'}-relevant.get(typ,set())
        if unexpected:raise ValueError('unsupported keyword for schema type: '+sorted(unexpected)[0])
        if typ is None:return self.any
        if typ=='object':
            props=s.get('properties',{})
            required=s.get('required',[])
            if not isinstance(props,dict) or not isinstance(required,list) or not all(isinstance(k,str) for k in required):
                raise ValueError('object properties/required have invalid types')
            if set(required)-set(props):raise ValueError('required property missing from properties')
            extra=s.get('additionalProperties',True)
            if extra is not False:
                if props or required or extra is not True:raise ValueError('unsupported open object properties: use additionalProperties=false')
                return self.object
            entries=[(k,self.value(v,depth+1)) for k,v in props.items()]
            cache={}
            def fields(i,previous):
                if (i,previous) in cache:return cache[i,previous]
                r=self.new();cache[i,previous]=r
                if i==len(entries):self.set(r,[[]]);return r
                k,v=entries[i]
                key=list(json.dumps(k,ensure_ascii=False).encode())
                take=([44,self.ws] if previous else [])+key+[self.ws,58,self.ws,v,fields(i+1,True)]
                self.set(r,[take]+([] if k in required else [[fields(i+1,previous)]]));return r
            return self.new([[123,self.ws,fields(0,False),125,self.ws]])
        if typ=='array':
            lo=s.get('minItems',0);hi=s.get('maxItems')
            if type(lo) is not int or lo<0 or lo>256 or (hi is not None and (type(hi) is not int or hi<lo or hi>256)):
                raise ValueError('unsupported array bounds: expected 0..256')
            item=self.value(s.get('items',{}),depth+1)
            if hi is None:
                more=self.new();self.set(more,[[],[44,self.ws,item,more]])
                if lo==0:seq=self.new([[],[item,more]])
                else:seq=self.new([[item]+[x for _ in range(lo-1) for x in [44,self.ws,item]]+[more]])
            else:
                seq=self.new([[x for i in range(n) for x in (([44,self.ws] if i else [])+[item])] for n in range(lo,hi+1)])
            return self.new([[91,self.ws,seq,93,self.ws]])
        primitives={'string':self.string,'integer':self.int_value,'number':self.number,'boolean':self.boolean,'null':self.null}
        if typ not in primitives:raise ValueError('unsupported schema type: '+str(typ))
        return primitives[typ]

def compile_schema(schema):
    c=Compiler();root=c.new([[c.ws,c.value(schema)]])
    return {'rules':c.rules,'root':root}

def write_bundle(path,grammar,tokens,eos,think_end,thinking):
    """Length-prefixed bytes preserve split UTF-8 tokens; None bans control/special tokens in the answer."""
    with Path(path).open('wb') as f:
        def ints(*v):f.write(struct.pack('<'+'i'*len(v),*v))
        f.write(b'SGR1');ints(len(tokens),eos,think_end,int(bool(thinking)),len(grammar['rules']),grammar['root'])
        for token in tokens:
            ints(-1 if token is None else len(token))
            if token is not None:f.write(token)
        for rule in grammar['rules']:
            ints(len(rule))
            for seq in rule:ints(len(seq));ints(*seq)

_cache=tempfile.TemporaryDirectory(prefix='strata-native-grammar-')
_vocab={}

class NativeValidator:
    native=True
    def __init__(self,validator,grammar):self.inner=validator;self.grammar=grammar
    def iter_errors(self,value):return self.inner.iter_errors(value)

def prepare_format(fmt,messages,svc,req,with_tools=False):
    from serve.structured import prepare_format as legacy
    req.pop('_native_grammar',None)
    if not isinstance(fmt,dict) or fmt.get('type') not in ('json_object','json_schema'):
        return legacy(fmt,messages,with_tools=with_tools,require_constraint=svc.require_strict_schema_constraint)
    if svc.model_arch!='deepseek41':
        return legacy(fmt,messages,with_tools=with_tools,require_constraint=svc.require_strict_schema_constraint)
    svc.load()
    if getattr(svc.engine,'info',{}).get('json_schema_constraint')!=1:
        return legacy(fmt,messages,with_tools=with_tools,require_constraint=svc.require_strict_schema_constraint)
    if with_tools:raise ValueError('native structured responses with tools are unsupported')
    if fmt.get('repair_latex_escapes'):raise ValueError('native constraints do not repair output')
    _,validator=legacy(fmt,messages,with_tools=with_tools,require_constraint=False)
    schema={'type':'object'} if fmt['type']=='json_object' else fmt['json_schema']['schema']
    return messages,NativeValidator(validator,compile_schema(schema))

def bind_request(req,validator,tok,thinking):
    if not getattr(validator,'native',False):return
    if id(tok) not in _vocab:
        end=tok.ids.get('</think>')
        if end is None:raise ValueError('native constraint requires a single thinking boundary token')
        types=tok.token_types or []
        tokens=[None if i<len(types) and types[i] in (3,4) else tok.token_bytes(i) for i in range(len(tok.tokens))]
        _vocab[id(tok)]=(tokens,tok.special_ids['eos'],end)
    tokens,eos,end=_vocab[id(tok)]
    key=hashlib.sha256(json.dumps([id(tok),validator.grammar,bool(thinking)],sort_keys=True).encode()).hexdigest()
    path=Path(_cache.name)/(key+'.bin')
    if not path.exists():write_bundle(path,validator.grammar,tokens,eos,end,thinking)
    req['_native_grammar']=str(path)

def validated_raw(text,validator,finish):
    from serve.structured import StructuredOutputError
    if finish!='stop':raise StructuredOutputError('native constrained JSON did not complete: '+str(finish))
    try:value=json.loads(text)
    except (ValueError,TypeError) as exc:raise StructuredOutputError('native constrained output is not complete JSON') from exc
    errors=list(validator.iter_errors(value))
    if errors:raise StructuredOutputError('native constrained output failed schema validation: '+errors[0].message)
    return text

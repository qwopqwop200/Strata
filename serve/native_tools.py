"""Per-function DSML argument constraints, using the native byte CFG sampler.

Only explicit strict=true enables schema masks. The envelope is structural for every
function; non-strict argument values are free text, without a schema constraint.
No prompt edits, output repair, validation retries, or separate sampling engine.
"""
import hashlib
import json
import re
from pathlib import Path
from serve import native_schema

MARK='<｜DSML｜'
END_PARAM='</｜DSML｜ parameter>'

def compile_tools(tools, choice=None):
    from jsonschema import Draft202012Validator
    from serve.frontend import tool_choice_of
    c=native_schema.Compiler(dsml=True)
    lit=lambda s:list(s.encode('utf-8'))
    # Raw DSML string values are escaped by the existing API parser. Excluding '<'
    # prevents data from becoming a protocol tag. Nested JSON uses \u003c instead.
    tail=c.rng(128,191)
    rawchar=c.new([[c.rng(9,10)],[13],[c.rng(32,59)],[c.rng(61,127)],
        [c.rng(194,223),tail],[224,c.rng(160,191),tail],
        [c.rng(225,236),tail,tail],[237,c.rng(128,159),tail],
        [c.rng(238,239),tail,tail],[240,c.rng(144,191),tail,tail],
        [c.rng(241,243),tail,tail,tail],[244,c.rng(128,143),tail,tail]])
    raw=c.new();c.set(raw,[[],[rawchar,raw]])
    namechar=c.new([[c.rng(48,57)],[c.rng(65,90)],[c.rng(97,122)],[95],[45]])
    anyname=c.new();c.set(anyname,[[namechar],[namechar,anyname]])

    def typed_parameter(key,s):
        # Validate all keywords even if the enum/const determines its serialization.
        c.value(s)
        values=s.get('enum',[s['const']] if 'const' in s else None)
        variants=[]
        if values is not None:
            for v in values:
                if isinstance(v,str):
                    if '<' in v:raise ValueError('unsupported DSML string enum containing <')
                    variants.append(('true',lit(v)))
                else:variants.append(('false',c.literal(v)))
        else:
            types=s.get('type')
            types=types if isinstance(types,list) else [types]
            for t in types:
                if t=='string':variants.append(('true',[raw]))
                else:variants.append(('false',[c.value({**s,'type':t})] if t else [c.any]))
        return c.new([lit(MARK+' parameter name="'+key+'" string="'+flag+'">')+body+lit(END_PARAM)+[c.ws]
                      for flag,body in variants])

    kind,named=tool_choice_of(choice)
    names=set();invokes=[]
    for t in tools:
        name=t.get('name')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',name):
            raise ValueError('unsupported DSML function name')
        if name in names:raise ValueError('duplicate tool function name: '+name)
        names.add(name)
        strict=t.get('strict',False)
        if type(strict) is not bool:raise ValueError('function.strict must be boolean')
        if strict:
            schema=t.get('parameters')
            try:Draft202012Validator.check_schema(schema)
            except Exception as exc:
                from jsonschema.exceptions import SchemaError
                if not isinstance(exc,SchemaError):raise
                raise ValueError('invalid strict tool schema: '+exc.message) from None
            c.value(schema)
            if schema.get('type')!='object' or schema.get('additionalProperties') is not False or 'enum' in schema or 'const' in schema:
                raise ValueError('unsupported strict tool root: use type object and additionalProperties=false without root enum/const')
            props=schema.get('properties',{});required=schema.get('required',[])
            entries=[]
            for key,s in props.items():
                if not re.fullmatch(r'[A-Za-z0-9_가-힣.-]{1,128}',key):
                    raise ValueError('unsupported DSML parameter name')
                entries.append((key,typed_parameter(key,s)))
            fields=c.new([[]])
            for key,param in reversed(entries):
                fields=c.new([[param,fields]]+([] if key in required else [[fields]]))
        else:
            # Only DSML framing is constrained here. Values need not match the
            # non-strict function's parameters (they need not even be JSON).
            param=c.new([lit(MARK+' parameter name="')+[anyname]+lit('" string="'+flag+'">')+[raw]+lit(END_PARAM)+[c.ws]
                         for flag in ('true','false')])
            fields=c.new();c.set(fields,[[],[param,fields]])
        invoke=c.new([lit(MARK+' invoke name="'+name+'">')+[c.ws,fields]+lit('</｜DSML｜ invoke>')+[c.ws]])
        if kind!='named' or name==named:invokes.append([invoke])
    if not invokes:raise ValueError('no function available for tool_choice')
    invoke=c.new(invokes)
    more=c.new();c.set(more,[lit('</｜DSML｜ calls>'),[invoke,more]])
    root=c.new([lit(' calls>')+[c.ws,invoke,more]])
    return {'rules':c.rules,'root':root,'required':kind in ('named','required')}

def prepare(req,tools,svc):
    # Private transport keys are never trusted from clients, including tool_choice=none.
    req.pop('_native_tools',None)
    if any('strict' in t and type(t['strict']) is not bool for t in tools or []):
        raise ValueError('function.strict must be boolean')
    if not any(t.get('strict') is True for t in tools or []):return
    if req.get('response_format') and req['response_format'].get('type') not in (None,'text'):
        raise ValueError('strict tools with structured response_format are unsupported')
    if svc.model_arch!='deepseek41':raise ValueError('strict tool constraints require the native DeepSeek engine')
    grammar=compile_tools(tools,req.get('tool_choice'))
    svc.load()
    info=getattr(svc.engine,'info',{})
    if info.get('strict_tool_constraint')!=1 or info.get('full_vocab_sampling')!=1:
        raise ValueError('this engine does not support native strict tool constraints')
    if req.get('strata_mcp'):raise ValueError('strict tool constraints with MCP are unsupported')
    req['_native_tools']=grammar

def bind_segment(sampling,tok,prefix):
    """Fresh grammar per GEN, replaying only this answer's emitted/injected tokens.
    Prefix KV restores and speculative overrun must never supply grammar state.
    """
    grammar=sampling.get('_native_tools')
    if grammar is None:return sampling
    key=('tools',id(tok))
    if key not in native_schema._vocab:
        types=tok.token_types or []
        tokens=[None if i<len(types) and types[i] in (3,4) and tok.tokens[i]!='｜DSML｜' else tok.token_bytes(i)
                for i in range(len(tok.tokens))]
        native_schema._vocab[key]=(tokens,tok.special_ids['eos'],tok.ids['</think>'])
    tokens,eos,end=native_schema._vocab[key]
    digest=hashlib.sha256(json.dumps([id(tok),grammar,list(prefix)],sort_keys=True).encode()).hexdigest()
    path=Path(native_schema._cache.name)/(digest+'.bin')
    if not path.exists():native_schema.write_bundle(path,grammar,tokens,eos,end,grammar.get('required',False),tool_prefix=prefix)
    return {**sampling,'_native_grammar':str(path)}

def release_segment(sampling):
    if sampling.get('_native_tools') and sampling.get('_native_grammar'):
        path=Path(sampling['_native_grammar'])
        if path.parent==Path(native_schema._cache.name):path.unlink(missing_ok=True)

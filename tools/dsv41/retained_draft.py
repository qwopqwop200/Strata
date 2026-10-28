"""Metadata-only Q4 drafter reattachment. Never rewrite a target or draft payload."""
from pathlib import Path
import hashlib,json,os

def read_table(path):
    files, tensors = {}, {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f: continue
        if f[0] == 'file': files[int(f[1])] = f[2]
        elif f[0] == 't':
            if f[1] in tensors: raise ValueError('duplicate tensor '+f[1])
            tensors[f[1]] = f
    return files, tensors

def attach_q4(pack, model, retained):
    pack, model, retained = map(lambda p:Path(p).resolve(), (pack,model,retained))
    retention=json.loads((retained/'retention-manifest.json').read_text())
    if retention.get('status')!='RETAINED_VERIFIED': raise ValueError('verified retained draft required')
    descriptor=json.loads((retained/'historical-draft-quant.json').read_text())
    if descriptor.get('quant')!='Q4_K' or descriptor.get('lossy_draft_only')is not True: raise ValueError('original selected Q4 draft descriptor required')
    draft=retained/'draft.bin'; record=next(a for a in retention['artifacts']if Path(a['destination']).resolve()==draft)
    st=draft.stat()
    if (st.st_dev,st.st_ino,st.st_size)!=(record['device'],record['inode'],record['size']) or record['sha256']!=descriptor['artifact']['sha256']:raise ValueError('retained draft identity differs from verified record')
    _, old=read_table(retained/'historical-pack/maya-ds4-q4/tensors.txt')
    original_text=(pack/'tensors.txt').read_text();files,tensors=read_table(pack/'tensors.txt');original_files=dict(files);original={n:tuple(t)for n,t in tensors.items()if not n.startswith('mtp.')}
    selected=[c['name']for c in descriptor['converted']]
    if len(selected)!=len(set(selected)) or not selected or any(not n.startswith('mtp.')for n in selected): raise ValueError('draft-only unique converted names required')
    rel=os.path.relpath(draft,model).replace(os.sep,'/')
    if any(c.isspace()for c in rel):raise ValueError('unsupported whitespace in path')
    fi=max(files,default=-1)+1;files[fi]=rel
    for n in selected:
        if n not in old or n not in tensors:raise ValueError('missing selected draft tensor '+n)
        row=old[n];current=tensors[n];off,size=int(row[4]),int(row[5])
        if row[2]!='Q4_K' or row[6:]!=current[6:] or off<0 or size<=0 or off+size>st.st_size:raise ValueError('retained draft type/shape/range mismatch '+n)
        tensors[n]=[row[0],row[1],row[2],str(fi),*row[4:]]
    if original!={n:tuple(t)for n,t in tensors.items()if not n.startswith('mtp.')}:raise ValueError('Atomic target mapping changed')
    # Preserve every original file/header/comment/nonselected tensor byte.
    # Only the selected mtp rows are replaced; insert one new file record.
    lines=[];inserted=False
    for line in original_text.splitlines(keepends=True):
        fields=line.split()
        if fields and fields[0]=='t' and not inserted:
            lines.append(f'file {fi} {rel}\n');inserted=True
        if fields and fields[0]=='t' and fields[1]in selected:
            lines.append(' '.join(tensors[fields[1]])+'\n')
        else:lines.append(line)
    if not inserted:raise ValueError('no tensor lines')
    raw=''.join(lines)
    (pack/'tensors.txt').write_text(raw)
    after_files,after_tensors=read_table(pack/'tensors.txt')
    if any(after_files.get(i)!=name for i,name in original_files.items())or original!={n:tuple(t)for n,t in after_tensors.items()if not n.startswith('mtp.')}:raise ValueError('original file/non-MTP metadata changed')
    proof=dict(schema=1,scope='retained_selected_draft_only',draft=str(draft),draft_sha256=record['sha256'],payload_hash_verification='parent retained manifest; metadata-only operation did not rehash8GB',retention_manifest_sha256=hashlib.sha256((retained/'retention-manifest.json').read_bytes()).hexdigest(),converted_count=len(selected),converted_names=selected,converted_bounds_verified=True,converted_dtype='Q4_K',draft_bytes=st.st_size,converted_tensors={n:dict(dtype=tensors[n][2],offset=int(tensors[n][4]),bytes=int(tensors[n][5]),shape=list(map(int,tensors[n][6:])))for n in selected},target_non_mtp_mapping_unchanged=True,target_weight_payload_modified=False,draft_payload_modified=False,target_mapping_sha256=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    (pack/'retained-draft-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof

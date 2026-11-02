"""Prepare and load artifacts produced by registered adapters on local data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .registry import create, resolve


def load_matrix(config, spec):
    path=resolve(config,spec['path'])
    if spec.get('format','npy')=='npy':
        rows=np.load(path,mmap_mode='r',allow_pickle=False)
    elif spec['format']=='f32':
        shape=tuple(spec['shape'])
        if len(shape)!=2 or min(shape)<1 or path.stat().st_size!=int(np.prod(shape))*4:
            raise ValueError(f'Invalid float32 matrix shape or file size: {path}')
        rows=np.memmap(path,dtype='<f4',mode='r',shape=shape)
    else: raise ValueError('Matrix format must be npy or f32')
    if rows.ndim!=2 or min(rows.shape)<1 or not np.issubdtype(rows.dtype,np.number):
        raise ValueError('Expected a nonempty numeric [series, dimensions] matrix')
    return rows


def finite_chunks(rows, batch):
    for start in range(0,len(rows),batch):
        chunk=np.asarray(rows[start:start+batch])
        if not np.isfinite(chunk).all(): raise ValueError('Series contain nonfinite values')
        yield start, chunk


def dataset_entry(config, name):
    return next(e for e in config['datasets'] if e['id']==name)


def artifact_dir(config,dataset,method,index=None):
    path=resolve(config,config.get('artifacts','artifacts'))/dataset/method
    return path/'indexes'/index if index else path


def signature(config, dataset, method, index=None):
    entry=dataset_entry(config,dataset)
    files=[resolve(config,entry['data']['path'])]
    if 'queries' in entry: files.append(resolve(config,entry['queries']['path']))
    details={'dataset':entry,'method':next(e for e in config['methods'] if e['id']==method),
             'files':[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]}
    if index: details['index']=next(e for e in config['indexes'] if e['id']==index)
    return hashlib.sha256(json.dumps(details,sort_keys=True).encode()).hexdigest()


def prepare(config, datasets=None, methods=None, indexes=None, force=False):
    batch=int(config.get('batch_size',4096))
    if batch<1: raise ValueError('batch_size must be positive')
    for ds in config['datasets']:
        if datasets and ds['id'] not in datasets: continue
        data=load_matrix(config,ds['data'])
        for _,_chunk in finite_chunks(data,batch): pass
        if 'queries' in ds:
            q=ds['queries']
            if q.get('kind','ids')=='series':
                queries=load_matrix(config,q)
                if queries.shape[1]!=data.shape[1]: raise ValueError('Query dimension mismatch')
                for _,_chunk in finite_chunks(queries,batch): pass
            else:
                ids=np.load(resolve(config,q['path']),allow_pickle=False)
                if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)==0 or np.any(ids<0) or np.any(ids>=len(data)):
                    raise ValueError('Query IDs must be valid integer data row IDs')
        for method in config['methods']:
            mid=method['id']
            if methods and mid not in methods: continue
            out=artifact_dir(config,ds['id'],mid); out.mkdir(parents=True,exist_ok=True)
            manifest=out/'manifest.json'; fingerprint=signature(config,ds['id'],mid)
            ready=manifest.exists() and (out/'representations.npy').exists()
            ready=ready and json.loads(manifest.read_text())['signature']==fingerprint
            plugin=create(method)
            if not ready or force:
                manifest.unlink(missing_ok=True)
                plugin.fit(data,out)
                target=None
                for start,chunk in finite_chunks(data,batch):
                    encoded=np.asarray(plugin.transform(chunk),dtype=np.float32)
                    if encoded.shape!=(len(chunk),int(method['dimensions'])) or not np.isfinite(encoded).all():
                        raise ValueError(f'Invalid representation matrix from {mid}')
                    if target is None:
                        target=np.lib.format.open_memmap(out/'representations.npy',mode='w+',dtype=np.float32,shape=(len(data),encoded.shape[1]))
                    target[start:start+len(chunk)]=encoded
                target.flush(); del target
                manifest.write_text(json.dumps({'signature':fingerprint,'shape':[len(data),method['dimensions']]}))
            else: plugin.load(out)
            representations=np.load(out/'representations.npy',mmap_mode='r',allow_pickle=False)
            for idx in config['indexes']:
                if indexes and idx['id'] not in indexes: continue
                directory=artifact_dir(config,ds['id'],mid,idx['id']); directory.mkdir(parents=True,exist_ok=True)
                marker=directory/'manifest.json'; sig=signature(config,ds['id'],mid,idx['id'])
                if force or not marker.exists() or json.loads(marker.read_text())['signature']!=sig:
                    marker.unlink(missing_ok=True)
                    create(idx).build(data,representations,directory)
                    marker.write_text(json.dumps({'signature':sig}))
                print(f"Prepared {ds['id']} / {mid} / {idx['id']}",flush=True)

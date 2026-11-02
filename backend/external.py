"""Adapters connecting registered artifacts to the visualization workspaces."""
from time import perf_counter
import json
import numpy as np
from .contracts import Approximation, SearchResult
from .plugins import IndexRequest
from .registry import create, read_config, resolve
from .pipeline import artifact_dir, dataset_entry, load_matrix, signature


class ExternalStore:
    def __init__(self, config=None):
        self.config=config or read_config()

    def data(self,name):
        return load_matrix(self.config,dataset_entry(self.config,name)['data'])

    def query_ids(self,name,count=None):
        ds=dataset_entry(self.config,name)
        q=ds.get('queries')
        if q is None: ids=list(range(len(self.data(name))))
        elif q.get('kind','ids')=='series': ids=[-i-1 for i in range(len(load_matrix(self.config,q)))]
        else:
            values=np.load(resolve(self.config,q['path']),allow_pickle=False)
            if values.ndim!=1 or not np.issubdtype(values.dtype,np.integer) or len(values)==0 or np.any(values<0) or np.any(values>=len(self.data(name))):
                raise ValueError('Query IDs must reference valid data rows')
            ids=values.tolist()
        return ids if count is None else ids[:count]

    def query(self,name,qid):
        if qid>=0: return self.data(name)[qid]
        return load_matrix(self.config,dataset_entry(self.config,name)['queries'])[-qid-1]


class ExternalModelProvider:
    def __init__(self,store): self.store=store; self._loaded={}

    def list_models(self):
        return [dict(description='',group='Data-adaptive',**entry) for entry in self.store.config['methods']]

    def plugin(self,name,mid):
        key=(name,mid)
        if key not in self._loaded:
            config=self.store.config; directory=artifact_dir(config,name,mid)
            marker=directory/'manifest.json'
            if not marker.exists() or json.loads(marker.read_text())['signature']!=signature(config,name,mid):
                raise RuntimeError(f'Run prepare.py for {name} / {mid} before opening this configuration')
            plugin=create(next(e for e in config['methods'] if e['id']==mid))
            plugin.load(directory); self._loaded[key]=plugin
        return self._loaded[key]

    def approximate(self,series,model_id,segments,cardinality,name):
        result=self.plugin(name,model_id).approximate(np.asarray(series),cardinality)
        entry=next(e for e in self.store.config['methods'] if e['id']==model_id)
        if not isinstance(result,Approximation): raise TypeError('approximate must return Approximation')
        if np.asarray(result.coefficients).shape!=(int(entry['dimensions']),) or not np.isfinite(result.coefficients).all():
            raise ValueError('Query representation shape or values are invalid')
        return result


class ExternalIndexProvider:
    is_external=True

    def __init__(self,store,models): self.store=store; self.models=models; self._contexts={}

    def prepare_context(self,name,model_id,index_type):
        key=(name,model_id,index_type)
        if key in self._contexts: return self._contexts[key]
        config=self.store.config; self.models.plugin(name,model_id)
        directory=artifact_dir(config,name,model_id,index_type); marker=directory/'manifest.json'
        if not marker.exists() or json.loads(marker.read_text())['signature']!=signature(config,name,model_id,index_type):
            raise RuntimeError(f'Run prepare.py for {name} / {model_id} / {index_type}')
        representations=np.load(artifact_dir(config,name,model_id)/'representations.npy',mmap_mode='r',allow_pickle=False)
        plugin=create(next(e for e in config['indexes'] if e['id']==index_type))
        plugin.load(self.store.data(name),representations,directory)
        description=plugin.describe()
        nodes=description['nodes']
        required={'id','label','depth','parent','children','ids'}
        if any(not required.issubset(n) for n in nodes): raise ValueError('Index node schema is incomplete')
        if not nodes: raise ValueError('Index describe() must return at least one node')
        for n in nodes:
            if n['parent'] is not None and n['parent']>=n['id']:
                raise ValueError('Tree node IDs must place parents before children')
        context=dict(name=name,model_id=model_id,coefficients=representations,nodes=nodes,
                     external_index=description,plugin=plugin)
        self._contexts[key]=context
        return context

    def ground_truth(self,name,query,query_id,k):
        return exact_neighbors(query,self.store.data(name),k,query_id)

    def representation_knn(self,context,query,k,query_id):
        return exact_neighbors(query,context['coefficients'],k,query_id)

    def search_curve(self,query,context,k,exclude_id,budgets,representation,encoding_ms):
        request=IndexRequest(query,representation,self.store.data(context['name']),k,exclude_id,
                             tuple(float(b) for b in budgets),encoding_ms)
        start=perf_counter(); results=context['plugin'].search_curve(request)
        for budget in budgets:
            result=results[float(budget)]
            if not isinstance(result,SearchResult): raise TypeError('search_curve must return SearchResult snapshots')
            if len(result.ids)!=len(result.distances) or len(result.ids)>k or len(set(result.ids))!=len(result.ids):
                raise ValueError('Search IDs must be unique and aligned with distances')
            if any(not isinstance(i,(int,np.integer)) or i<0 or i>=len(request.data) or i==exclude_id for i in result.ids):
                raise ValueError('Search returned invalid row IDs')
            if not np.isfinite(result.distances).all(): raise ValueError('Nonfinite search distances')
            if 'estimated_ms' not in result.metrics: raise ValueError('Snapshot must report measured elapsed time as estimated_ms')
            result.metrics['encoding_ms']=encoding_ms
            result.metrics['adapter_wall_ms']=(perf_counter()-start)*1000
            result.metrics.setdefault('candidates',0)
        return results


def exact_neighbors(query,rows,k,exclude_id=None,batch_size=8192):
    """Streaming Euclidean reference evaluation with bounded temporary memory."""
    distances=np.empty(0,dtype=np.float64); ids=np.empty(0,dtype=np.int64)
    for start in range(0,len(rows),batch_size):
        block=np.asarray(rows[start:start+batch_size],dtype=np.float64)
        values=np.linalg.norm(block-np.asarray(query,dtype=np.float64),axis=1)
        local=np.arange(start,start+len(block))
        if exclude_id is not None: values[local==exclude_id]=np.inf
        distances=np.concatenate([distances,values]); ids=np.concatenate([ids,local])
        order=np.lexsort((ids,distances))[:k]
        distances,ids=distances[order],ids[order]
    return ids[np.isfinite(distances)].astype(int).tolist()

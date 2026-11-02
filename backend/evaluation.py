"""Reference evaluation of registered methods and measured index snapshots."""
from time import perf_counter
import numpy as np
from .external import exact_neighbors

BUDGETS=(10.,25.,50.,100.,200.,500.,1000.)

def query_ids(size,total=100):
    return list(range(min(size,total)))

def match_neighbors(returned, truth, data, epsilon=1e-6):
    """Return assignments in result order; deduplicate repeated returned IDs."""
    if epsilon <= 0:
        raise ValueError("Tolerance must be positive")
    adjacency, seen = [], set()
    for rid in returned:
        edges = [j for j, gid in enumerate(truth) if np.linalg.norm(data[rid]-data[gid]) < epsilon] if rid not in seen else []
        # Prefer identical IDs when possible; augmenting paths still maximize count.
        edges.sort(key=lambda j: truth[j] != rid)
        adjacency.append(edges)
        seen.add(rid)
    assigned = {}
    def augment(i, visited):
        for j in adjacency[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or augment(assigned[j], visited):
                assigned[j] = i
                return True
        return False
    for i in range(len(returned)):
        augment(i, set())
    inverse = {i: j for j,i in assigned.items()}
    unmatched_results=[i for i in range(len(returned)) if i not in inverse]
    unmatched_truth=[j for j in range(len(truth)) if j not in assigned]
    # Pair remaining results and truth items only for miss diagnostics. These
    # pairs do not affect the maximum-matching score above.
    diagnostic={}
    while unmatched_results and unmatched_truth:
        _,ri,tj=min((float(np.linalg.norm(data[returned[i]]-data[truth[j]])),i,j)
                    for i in unmatched_results for j in unmatched_truth)
        diagnostic[ri]=tj
        unmatched_results.remove(ri)
        unmatched_truth.remove(tj)
    rows = []
    for i,rid in enumerate(returned):
        matched=i in inverse
        truth_index=inverse[i] if matched else diagnostic.get(i)
        gid=truth[truth_index] if truth_index is not None else None
        rows.append(dict(rank=i+1, result_id=rid, truth_id=gid,
                         status="ID match" if matched and gid == rid else "Tolerance match" if matched else "Miss",
                         distance=None if gid is None else float(np.linalg.norm(data[rid]-data[gid]))))
    return dict(score=len(assigned)/len(truth) if truth else 0., matches=rows,
                missed=[gid for j,gid in enumerate(truth) if j not in assigned])



def evaluate_query(query,query_id,data,contexts,models,indexes,segments,cardinality,top_k,epsilon,budgets=BUDGETS,dataset_name=None,compact=False):
    exclude_id=query_id if query_id is not None and query_id>=0 else None
    truth=indexes.ground_truth(dataset_name,query,exclude_id,top_k)
    methods={}
    for mid,context in contexts.items():
        started=perf_counter()
        approx=models.approximate(query,mid,segments,cardinality,dataset_name)
        encoding_ms=(perf_counter()-started)*1000
        neighbors=indexes.representation_knn(context,approx.coefficients,top_k,exclude_id)
        coverage=match_neighbors(neighbors,truth,data,epsilon)
        results=indexes.search_curve(query,context,top_k,exclude_id,budgets,approx.coefficients,encoding_ms)
        searches={}
        for budget in budgets:
            result=results[float(budget)]
            matching=match_neighbors(result.ids,truth,data,epsilon)
            if compact: result.trace=[]; result.nodes=[]
            searches[budget]=dict(result=result,matching=matching)
        methods[mid]=dict(approximation=approx,neighbors=neighbors,coverage=coverage,searches=searches)
    return dict(query_id=query_id,query=query,ground_truth=truth,methods=methods)

def suite_rows(evaluations, names):
    rows=[]
    for evaluation in evaluations:
        for mid, entry in evaluation["methods"].items():
            for budget, search in entry["searches"].items():
                result=search["result"]
                rows.append(dict(query_id=evaluation["query_id"],method_id=mid,method=names[mid],budget_ms=budget,
                    coverage=entry["coverage"]["score"],recall=search["matching"]["score"],
                    candidates=result.metrics["candidates"],elapsed_ms=result.metrics["estimated_ms"],
                    inference_ms=result.metrics.get("encoding_ms",0.),io_ms=result.metrics.get("io_ms",0.),
                    leaves_read=result.metrics.get("leaves_read",0),pruned_leaves=result.metrics.get("pruned_leaves",0),
                    pruned_series=result.metrics.get("pruned_series",0),
                    series_excluded=result.metrics.get("series_excluded_by_lb",result.metrics.get("pruned_series",0)),
                    lb_checks=result.metrics.get("lb_checks",result.metrics.get("visited",0))))
    return rows

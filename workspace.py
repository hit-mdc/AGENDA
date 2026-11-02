"""Comparison views; model and index work stays behind the provider contracts."""
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from backend.evaluation import BUDGETS, evaluate_query, suite_rows, query_ids
from ui import chart_layout, series_chart, tree_chart, METHOD_COLORS, METHOD_MARKERS, METHOD_DASHES

RESULT_VERSION="2026-09-21-interface-taxonomy-v3"


@st.cache_data(show_spinner=False,max_entries=32)
def tsne_coordinates(features):
    """Small deterministic exact t-SNE for the visual sample (NumPy only).

    Keeping this local avoids a large compiled dependency in the conference
    demo.  The input is capped at a few hundred rows by the caller.
    """
    features=np.asarray(features,dtype=np.float32)
    features=features-features.mean(axis=0,keepdims=True)
    scale=np.maximum(features.std(axis=0,keepdims=True),1e-6)
    features=features/scale
    # PCA initialization, as used by common t-SNE implementations.
    _,_,vt=np.linalg.svd(features,full_matrices=False)
    y=(features@vt[:2].T).astype(np.float64)
    y=y/np.maximum(y.std(),1e-9)*1e-4
    d2=np.maximum(np.sum(features*features,axis=1)[:,None]+np.sum(features*features,axis=1)[None,:]-2*features@features.T,0)
    perplexity=min(30,max(5,(len(features)-1)//4)); target=np.log(perplexity)
    p=np.zeros_like(d2,dtype=np.float64)
    for i in range(len(features)):
        lo,hi=-20.,20.
        for _ in range(16):
            beta=np.exp((lo+hi)/2)
            row=np.exp(-d2[i]*beta); row[i]=0.; row/=max(row.sum(),1e-12)
            entropy=-np.sum(row[row>0]*np.log(row[row>0]))
            if entropy>target: lo=(lo+hi)/2
            else: hi=(lo+hi)/2
        p[i]=row
    p=np.maximum((p+p.T)/(2*len(features)),1e-12)
    velocity=np.zeros_like(y); gains=np.ones_like(y)
    for iteration in range(60):
        delta=y[:,None,:]-y[None,:,:]
        inv=1/(1+np.sum(delta*delta,axis=2)); np.fill_diagonal(inv,0)
        q=np.maximum(inv/inv.sum(),1e-12)
        exaggeration=12 if iteration<100 else 1
        grad=4*np.sum(((exaggeration*p-q)*inv)[:,:,None]*delta,axis=1)
        momentum=.5 if iteration<100 else .8
        gains=np.where(np.sign(grad)!=np.sign(velocity),gains+.2,gains*.8)
        gains=np.maximum(gains,.01)
        velocity=momentum*velocity-200*gains*grad
        y+=velocity; y-=y.mean(axis=0,keepdims=True)
    return y.astype(np.float32)


def large_table(frame, formats=None):
    table=frame.copy()
    for column, formatter in (formats or {}).items():
        table[column]=table[column].map(formatter)
    st.markdown('<div class="large-table">'+table.to_html(index=False,escape=True,border=0)+'</div>',unsafe_allow_html=True)


def render_workspace(page, data, selected, models, indexes, context_builder, config, names, display_only=False):
    name, segments, cardinality, index_type, top_k, epsilon = config

    query_dataset=name
    fixed_ids=indexes.store.query_ids(query_dataset) if getattr(indexes,"is_external",False) else query_ids(100,len(data))

    def run_suite(selected, count, budgets, progress=None):
        contexts=context_builder()
        evaluations=[]
        ids=fixed_ids[:count]
        count=len(ids)
        for position,i in enumerate(ids,1):
            evaluations.append(evaluate_query(indexes.store.query(name,i),i,data,contexts,models,indexes,segments,cardinality,top_k,epsilon,
                                               budgets=budgets,dataset_name=name,compact=True))
            if progress is not None:
                progress.progress(position/count,text=f"Evaluating shared query {position} of {count}")
        return evaluations

    if not selected:
        st.info("Select at least one approximation method in the sidebar.")
        return
    if page == "Query Set":
        left,middle,right=st.columns([2,2,1.5])
        count=left.slider("Shared query set · Number of queries",1,max(2,len(fixed_ids)),value=min(10,len(fixed_ids)),step=1,key="query_count")
        budget=middle.select_slider("Maximum search budget · milliseconds",list(BUDGETS),value=100.,key="suite_budget")
        active_budgets=tuple(b for b in BUDGETS if b<=budget)
        right.write("")
        run=right.button("Run comparison →",type="primary",width="stretch",disabled=display_only)
        if display_only:
            st.info(f"{name} is available for interface preview. Query execution is not connected yet.")
            return
        signature=(RESULT_VERSION,config,tuple(selected),count,budget)
        if run:
            with st.status(f"Preparing {len(selected)} method indices…",expanded=True) as status:
                progress=st.progress(0,text="Loading cached representations and indices")
                results=run_suite(tuple(selected),count,active_budgets,progress)
                progress.empty()
                status.update(label=f"Completed {count} queries across {len(selected)} methods",state="complete",expanded=False)
                st.session_state.suite=(signature,results)
        if st.session_state.get("suite",(None,None))[0]!=signature:
            st.info("Choose the dataset, methods, index, query count, and time budget, then run the comparison.")
            return
        evaluations=st.session_state.suite[1]
        frame=pd.DataFrame(suite_rows(evaluations,names))
        current=frame[frame.budget_ms==budget]
        st.caption(f"Same {count} queries · seed 42 · k = {top_k} · ε = {epsilon:g} · self-match excluded")
        a,b,c=st.columns(3)
        a.metric("Queries",count)
        b.metric("Compared methods",len(selected))
        c.metric("Maximum budget",f"{budget:g} ms")
        st.subheader("Comparison at a glance")
        summary=current.groupby("method_id",sort=False).agg(
            coverage=("coverage","mean"),recall=("recall","mean"),candidates=("candidates","mean"),
            leaves=("leaves_read","mean"),pruned=("pruned_leaves","mean"),elapsed=("elapsed_ms","mean"),
            excluded=("series_excluded","mean"),checks=("lb_checks","mean"),
            inference=("inference_ms","mean"),io=("io_ms","mean")).reindex(selected)
        display=pd.DataFrame({"Method":[names[mid] for mid in selected],"NN coverage@k":summary.coverage.values,"Recall@k":summary.recall.values,
                              "Inference (ms)":summary.inference.values,"I/O (ms)":summary.io.values,
                              "LB node checks":summary.checks.values,"Series excluded by LB":summary.excluded.values,
                              "Leaf pages read":summary.leaves.values,"Raw distances":summary.candidates.values,
                              "Query time (ms)":summary.elapsed.values})
        large_table(display,{"NN coverage@k":lambda v:f"{v:.2%}","Recall@k":lambda v:f"{v:.2%}",
            "Inference (ms)":lambda v:f"{v:.2f}","I/O (ms)":lambda v:f"{v:.2f}",
            "LB node checks":lambda v:f"{v:,.1f}","Series excluded by LB":lambda v:f"{v:,.0f}",
            "Leaf pages read":lambda v:f"{v:.1f}","Raw distances":lambda v:f"{v:,.1f}","Query time (ms)":lambda v:f"{v:.2f}"})

        st.subheader("01 / Representation quality")
        fig=go.Figure()
        for mid in selected:
            values=current[current.method_id==mid]
            fig.add_trace(go.Box(x=values.coverage,name=names[mid],orientation="h",boxpoints="all",jitter=.3,pointpos=0,
                marker=dict(color=METHOD_COLORS[mid],size=8,symbol=METHOD_MARKERS[mid]),line=dict(color=METHOD_COLORS[mid]),boxmean=True,
                customdata=values.query_id,hovertemplate="Query %{customdata}<br>Coverage %{x:.2%}<extra>%{fullData.name}</extra>"))
        chart_layout(fig,max(440,len(selected)*68+90))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="NN coverage@k",range=[-.05,1.05],tickformat=".0%")
        st.plotly_chart(fig,width="stretch",theme=None,key="coverage_distribution")
        st.caption("Each point is one query. Boxes show the distribution; the dashed line shows the mean.")

        st.subheader("02 / Retrieval up to the maximum time budget")
        fig=go.Figure()
        for mid in selected:
            curve=frame[frame.method_id==mid].groupby("budget_ms",sort=True).agg(recall=("recall","mean"),elapsed=("elapsed_ms","mean"))
            fig.add_trace(go.Scatter(x=curve.elapsed,y=curve.recall,name=names[mid],mode="lines+markers",
                line=dict(color=METHOD_COLORS[mid],width=3),marker=dict(size=10,symbol=METHOD_MARKERS[mid],color=METHOD_COLORS[mid])))
        chart_layout(fig,560)
        fig.update_layout(legend=dict(y=1.05,yanchor="bottom",entrywidth=.32,entrywidthmode="fraction"),margin=dict(t=45+35*((len(selected)+2)//3)))
        fig.update_xaxes(title="Total query time (ms)",type="log")
        fig.update_yaxes(title="Mean recall@k",range=[-.05,1.05],tickformat=".0%")
        st.plotly_chart(fig,width="stretch",theme=None,key="recall_time")
        st.caption(f"Each point comes from the same measured run stopped at {budget:g} ms. Time includes representation inference, disk I/O, {index_type} traversal, and raw-series distance evaluation.")

        st.subheader("03 / Find a query worth inspecting")
        metric=st.selectbox("Inspect metric",["NN coverage","Recall at selected budget"])
        field="coverage" if metric=="NN coverage" else "recall"
        pivot=current.pivot(index="method_id",columns="query_id",values=field).reindex(index=selected,columns=[e["query_id"] for e in evaluations])
        fig=go.Figure(go.Heatmap(z=pivot.values,x=[str(i) for i in pivot.columns],y=[names[i] for i in pivot.index],
            zmin=0,zmax=1,colorscale=[[0,"#EFF3F8"],[.5,"#6DBFC0"],[1,"#087C6D"]],colorbar=dict(tickformat=".0%"),
            hovertemplate=f"%{{y}}<br>Query %{{x}}<br>{metric} %{{z:.2%}}<extra></extra>"))
        chart_layout(fig,max(370,65*len(selected)+140))
        fig.update_xaxes(title="Query ID",type="category",nticks=min(count,15))
        fig.update_layout(margin=dict(l=210,r=30,t=20,b=65))
        st.plotly_chart(fig,width="stretch",theme=None,key="query_heatmap")
        left,right=st.columns(2)
        ref=left.selectbox("Reference method",selected,format_func=names.get,key="suite_reference")
        target=right.selectbox("Compare against reference",selected,index=1 if len(selected)>1 else 0,format_func=names.get,key="delta_target")
        delta=(pivot.loc[target]-pivot.loc[ref])
        fig=go.Figure(go.Bar(x=[str(i) for i in delta.index],y=delta.values,marker_color=["#009D85" if v>=0 else "#C95D58" for v in delta.values]))
        chart_layout(fig,350)
        fig.update_yaxes(title=f"{metric}<br>{names[target]} minus {names[ref]}",tickformat="+.0%",range=[-1.05,1.05])
        fig.update_xaxes(title="Query ID",type="category",nticks=min(count,15))
        st.plotly_chart(fig,width="stretch",theme=None,key="query_delta")
        def open_query(qid):
            st.session_state.drill_query=int(qid)
            st.session_state.drill_budget=budget
            st.session_state.page="Single Query"
            st.session_state.playing=False
        a,b=st.columns([3,1])
        qid=a.selectbox("Inspect any query individually",[e["query_id"] for e in evaluations],format_func=lambda i:f"TS-{i:04d}",key="query_pick")
        b.write("")
        b.button("Inspect query →",on_click=open_query,args=(qid,),type="primary",width="stretch")
        st.download_button("↓ Export per-query results",frame.to_csv(index=False),file_name="agenda-query-set.csv",mime="text/csv")
        st.download_button("↓ Export experiment settings",json.dumps(dict(dataset=name,methods=selected,queries=[e["query_id"] for e in evaluations],segments=segments,cardinality=cardinality,index=index_type,k=top_k,epsilon=epsilon,budgets_ms=BUDGETS,timing="inference + traversal + candidate evaluation",seed=42),indent=2),file_name="agenda-settings.json",mime="application/json")
    else:
        render_single(data,selected,models,indexes,context_builder,config,names,display_only=display_only)


def render_single(data,selected,models,indexes,context_builder,config,names,display_only=False):
    name,segments,cardinality,index_type,top_k,epsilon=config
    if "drill_query" in st.session_state:
        st.session_state.single_id=st.session_state.pop("drill_query")
        st.session_state.single_source="Dataset sample"
        st.session_state.single_budget=st.session_state.pop("drill_budget")
    query_dataset=name
    fixed_queries=indexes.store.query_ids(query_dataset) if getattr(indexes,"is_external",False) else query_ids(100,len(data))
    if st.session_state.get("single_id") not in fixed_queries: st.session_state.single_id=fixed_queries[0]
    if st.session_state.get("single_budget") not in BUDGETS: st.session_state.single_budget=100.
    c1,c2=st.columns(2)
    source=c1.selectbox("Query source",["Dataset sample","Upload CSV"],key="single_source")
    qid=int(c2.selectbox("Query series ID",fixed_queries,key="single_id",format_func=lambda i:f"TS-{i:04d}",disabled=source!="Dataset sample"))
    query=indexes.store.query(name,qid)
    exclude=qid
    if source=="Upload CSV":
        length=data.shape[1]
        file=st.file_uploader(f"{length} numeric rows · one column · no header",type="csv")
        if file is None:
            st.info("Upload a query to compare the selected methods.")
            return
        try:
            values=pd.read_csv(file,header=None).to_numpy(dtype=float)
            if values.shape!=(length,1) or not np.isfinite(values).all() or values.std()<1e-8:
                raise ValueError(f"Expected {length} finite values with nonzero variance.")
            query=values[:,0]
            exclude=None
        except Exception as exc:
            st.error(f"Invalid query: {exc}")
            return
    budget=st.select_slider("Maximum search budget · milliseconds",list(BUDGETS),key="single_budget")
    run=st.button("Run query →",type="primary",width="stretch",disabled=display_only)
    if display_only:
        st.info(f"{name} is available for interface preview. Query execution is not connected yet.")
        return
    signature=(RESULT_VERSION,config,tuple(selected),exclude,query.tobytes(),budget)
    if run:
        with st.spinner("Comparing representation neighbors and budgeted search…"):
            contexts=context_builder()
            st.session_state.single_evaluation=evaluate_query(query,exclude,data,contexts,models,indexes,segments,cardinality,top_k,epsilon,budgets=(budget,),dataset_name=name)
        st.session_state.single_signature=signature
        st.session_state.replay_step=0
        st.session_state.playing=False
    if st.session_state.get("single_signature")!=signature:
        st.info("Choose the query source, query, methods, index, and time budget, then run the query.")
        return
    evaluation=st.session_state.single_evaluation
    rows=[]
    for mid in selected:
        entry=evaluation["methods"][mid]
        search=entry["searches"][budget]
        matched=sum(row["status"]!="Miss" for row in search["matching"]["matches"])
        rows.append({"Method":names[mid],"NN coverage@k":entry["coverage"]["score"],"Recall@k":search["matching"]["score"],
                     "Inference":f'{search["result"].metrics.get("encoding_ms",0.):.2f} ms',"I/O":f'{search["result"].metrics.get("io_ms",0):.2f} ms',"Query time":f'{search["result"].metrics["estimated_ms"]:.2f} ms',
                     "LB node checks":search["result"].metrics.get("lb_checks",0),
                     "Series excluded by LB":f'{search["result"].metrics.get("series_excluded_by_lb",0):,}',
                     "Leaf pages read":search["result"].metrics.get("leaves_read",0),
                     "Raw distances":f'{search["result"].metrics.get("candidates",0):,}',
                     "Matched / returned":f'{matched} / {len(search["result"].ids)}'})
    large_table(pd.DataFrame(rows),{"NN coverage@k":lambda v:f"{v:.0%}","Recall@k":lambda v:f"{v:.0%}"})
    st.caption(f"Ground truth: original-space exact kNN · k = {top_k} · one-to-one matching at ε = {epsilon:g}")
    st.subheader("01 / Representation quality")
    representative=list(selected[:2])
    with st.form("signal_method_form",border=False):
        visual=st.multiselect("Compare up to three DSA methods",selected,
            default=representative if len(representative)==2 else selected[:2],
            format_func=names.get,max_selections=3,key="signal_methods")
        st.form_submit_button("Update representation comparison",width="stretch")
    st.plotly_chart(series_chart([("Original query",query,"#172B43",False,"solid")],300),width="stretch",theme=None,key="single_signal")
    if visual:
        coeff_fig=make_subplots(rows=1,cols=len(visual),subplot_titles=[f"{names[mid]} · {len(evaluation['methods'][mid]['approximation'].coefficients)}D" for mid in visual])
        for column,mid in enumerate(visual,1):
            coefficients=evaluation["methods"][mid]["approximation"].coefficients
            coeff_fig.add_trace(go.Scatter(x=np.arange(1,len(coefficients)+1),y=coefficients,mode="lines+markers",showlegend=False,
                line=dict(color=METHOD_COLORS[mid],dash=METHOD_DASHES[mid],width=3),marker=dict(symbol=METHOD_MARKERS[mid],size=9),
                hovertemplate="Dimension %{x}<br>Value %{y:.3f}<extra></extra>"),row=1,col=column)
            coeff_fig.update_xaxes(title="Representation dimension",dtick=2,row=1,col=column)
            coeff_fig.update_yaxes(title="Representation value" if column==1 else None,row=1,col=column)
        chart_layout(coeff_fig,370); coeff_fig.update_layout(margin=dict(t=85,b=55,l=45,r=25))
        st.plotly_chart(coeff_fig,width="stretch",theme=None,key="coefficients_16d")

        rng=np.random.default_rng(2026)
        # Keep the live interaction responsive: 48 shared background samples are
        # enough to show local neighborhood structure without blocking each
        # multiselect change on a large exact t-SNE optimization.
        background=rng.choice(len(data),min(48,len(data)),replace=False).astype(int).tolist()
        important=set(evaluation["ground_truth"]+[exclude] if exclude is not None and exclude>=0 else evaluation["ground_truth"])
        for mid in visual: important.update(evaluation["methods"][mid]["neighbors"])
        sample=list(dict.fromkeys(background+list(important)))
        panels=[("Original space",None)]+[(f"{names[mid]} space",mid) for mid in visual]
        tsne_fig=make_subplots(rows=1,cols=len(panels),subplot_titles=[p[0] for p in panels])
        contexts=context_builder()
        for column,(label,mid) in enumerate(panels,1):
            features=np.asarray(data[sample]) if mid is None else np.asarray(contexts[mid]["coefficients"][sample])
            query_feature=query if mid is None else evaluation['methods'][mid]['approximation'].coefficients
            features=np.vstack([features,query_feature])
            coords=tsne_coordinates(features)
            lookup={sid:i for i,sid in enumerate(sample)}
            tsne_fig.add_trace(go.Scatter(x=coords[:,0],y=coords[:,1],mode="markers",name="Background",showlegend=column==1,
                marker=dict(color="#D8DEE7",size=5),hoverinfo="skip"),row=1,col=column)
            gt=[i for i in evaluation["ground_truth"] if i in lookup]
            tsne_fig.add_trace(go.Scatter(x=[coords[lookup[i],0] for i in gt],y=[coords[lookup[i],1] for i in gt],mode="markers",name="Ground truth kNN",showlegend=column==1,
                marker=dict(symbol="circle-open",color="#D48A18",size=13,line=dict(width=2)),text=[f"TS-{i:06d}" for i in gt],hovertemplate="%{text}<extra>Ground truth</extra>"),row=1,col=column)
            if mid is not None:
                neighbors=evaluation["methods"][mid]["neighbors"]
                tsne_fig.add_trace(go.Scatter(x=[coords[lookup[i],0] for i in neighbors],y=[coords[lookup[i],1] for i in neighbors],mode="markers",name=f"{names[mid]} representation kNN",showlegend=True,
                    marker=dict(symbol=METHOD_MARKERS[mid],color=METHOD_COLORS[mid],size=11,line=dict(color="white",width=2)),
                    text=[f"TS-{i:06d}" for i in neighbors],hovertemplate=f"%{{text}}<extra>{names[mid]} representation neighbor</extra>"),row=1,col=column)
            qpos=coords[-1]
            tsne_fig.add_trace(go.Scatter(x=[qpos[0]],y=[qpos[1]],mode="markers",name="Query",showlegend=column==1,
                marker=dict(symbol="star",color="#172B43",size=18),hovertemplate="Selected query<extra>Query</extra>"),row=1,col=column)
            tsne_fig.update_xaxes(visible=False,row=1,col=column); tsne_fig.update_yaxes(visible=False,row=1,col=column)
        chart_layout(tsne_fig,470); tsne_fig.update_layout(hovermode="closest",margin=dict(t=90,b=30,l=20,r=20),legend=dict(y=1.22))
        st.plotly_chart(tsne_fig,width="stretch",theme=None,key="representation_tsne")
        st.caption("t-SNE is fitted separately in each space on the same sampled series. Coordinates are comparable only within a panel; NN coverage remains the quantitative measure.")
    st.subheader("02 / Inspect retrieved neighbors")
    mid=st.selectbox("Inspect method",selected,format_func=names.get,key="inspect_method")
    entry=evaluation["methods"][mid]
    search=entry["searches"][budget]
    matching=search["matching"]
    a,b,c=st.columns(3)
    a.metric("Matched / ground truth",f'{round(matching["score"]*top_k)} / {top_k}')
    b.metric("Tolerance-only matches",sum(r["status"]=="Tolerance match" for r in matching["matches"]))
    c.metric("Missing neighbors",len(matching["missed"]))
    if matching["matches"]:
        table=pd.DataFrame([{"Rank":r["rank"],"Result":f'TS-{r["result_id"]:04d}',"Ground truth":"—" if r["truth_id"] is None else f'TS-{r["truth_id"]:04d}',"Match type":r["status"],
            "Distance: result → query":f'{np.linalg.norm(data[r["result_id"]]-query):.4f}',
            "Distance: ground truth → query":"—" if r["truth_id"] is None else f'{np.linalg.norm(data[r["truth_id"]]-query):.4f}',
            f"Distance to ground truth (ε={epsilon:g})":"—" if r["distance"] is None else f'{r["distance"]:.2e}'} for r in matching["matches"]])
        large_table(table)
        rank=st.selectbox("Inspect returned series",range(len(matching["matches"])),format_func=lambda i:f'#{i+1} · TS-{matching["matches"][i]["result_id"]:04d}',key="returned_rank")
        row=matching["matches"][rank]
        comparison=[("Query",query,"#172B43",False,"solid"),("Returned",data[row["result_id"]],METHOD_COLORS[mid],False,"dash")]
        if row["truth_id"] is not None:
            comparison.append(("Matched ground truth" if row["status"]!="Miss" else "Compared ground truth",data[row["truth_id"]],"#D48A18",False,"dot"))
        st.plotly_chart(series_chart(comparison,360),width="stretch",theme=None,key="match_signal")
        result_distance=np.linalg.norm(data[row["result_id"]]-query)
        truth_distance=None if row["truth_id"] is None else np.linalg.norm(data[row["truth_id"]]-query)
        st.caption(f'Distance: returned → query = {result_distance:.4f} · Distance: ground truth → query = {truth_distance:.4f}' if truth_distance is not None else f'Distance: returned → query = {result_distance:.4f}')
    else:
        st.info("No result fits within this budget. Increase the shared budget to inspect retrieved series; recall is 0%.")
    if matching["missed"]:
        missed=st.selectbox("Inspect a missed ground-truth neighbor",matching["missed"],format_func=lambda i:f"TS-{i:04d}",key="missed_id")
        paired=next((r for r in matching["matches"] if r["status"]=="Miss" and r["truth_id"]==missed),None)
        if paired is None and matching["matches"]:
            paired=min(matching["matches"],key=lambda r:np.linalg.norm(data[r["result_id"]]-data[missed]))
        miss_lines=[("Query",query,"#172B43",False,"solid")]
        if paired is not None:
            miss_lines.append(("Returned",data[paired["result_id"]],METHOD_COLORS[mid],False,"dash"))
        miss_lines.append(("Missed ground truth",data[missed],"#C95D58",False,"dot"))
        st.plotly_chart(series_chart(miss_lines,350),width="stretch",theme=None,key="missed_signal")
        truth_to_query=np.linalg.norm(data[missed]-query)
        if paired is not None:
            result_to_query=np.linalg.norm(data[paired["result_id"]]-query)
            st.caption(f'Distance: returned TS-{paired["result_id"]:06d} → query = {result_to_query:.4f} · Distance: ground truth TS-{missed:06d} → query = {truth_to_query:.4f} · Distance: returned → ground truth = {np.linalg.norm(data[paired["result_id"]]-data[missed]):.2e}')
        else:
            st.caption(f'Ground truth TS-{missed:04d} → query: {truth_to_query:.4f} · No series was returned within this budget.')
    with st.expander("Original-space ground truth"):
        large_table(pd.DataFrame({"Rank":range(1,top_k+1),"Series":[f"TS-{i:04d}" for i in evaluation["ground_truth"]],"Distance to query":[f"{np.linalg.norm(data[i]-query):.6f}" for i in evaluation["ground_truth"]]}))
    st.subheader("03 / Explore index and query process")
    reference=selected[0]
    compare_options=list(selected[1:]) or [reference]
    if st.session_state.get("second_traversal") not in compare_options:
        st.session_state.second_traversal=compare_options[0]
    compare=st.selectbox(f"Choose a DSA method for comparison. {names[reference]} is used as the reference.",compare_options,index=0,format_func=names.get,key="second_traversal")
    pair=list(dict.fromkeys([reference,compare]))
    total=max(len(evaluation["methods"][m]["searches"][budget]["result"].trace) for m in pair)
    replay_signature=(signature,budget,tuple(pair))
    if st.session_state.get("replay_signature")!=replay_signature:
        st.session_state.replay_signature=replay_signature
        st.session_state.replay_step=0
        st.session_state.playing=False
    @st.dialog("Index and query process",width="large")
    def replay_dialog():
        buttons=st.columns(4)
        if buttons[0].button("Play / pause",disabled=not total,width="stretch",key="dialog_play"):
            st.session_state.playing=not st.session_state.get("playing",False)
            if st.session_state.replay_step>=total: st.session_state.replay_step=0
        if buttons[1].button("← Back",width="stretch",key="dialog_back"):
            st.session_state.playing=False
            st.session_state.replay_step=max(0,st.session_state.replay_step-1)
        if buttons[2].button("Step →",width="stretch",key="dialog_step"):
            st.session_state.playing=False
            st.session_state.replay_step=min(total,st.session_state.replay_step+1)
        if buttons[3].button("Reset",width="stretch",key="dialog_reset"):
            st.session_state.playing=False
            st.session_state.replay_step=0
        def scrub_to_event():
            st.session_state.playing=False
            st.session_state.replay_step=st.session_state.replay_scrubber
        st.session_state.replay_scrubber=st.session_state.replay_step
        st.slider(f"Global traversal step · longest trace = {total}",0,total,key="replay_scrubber",on_change=scrub_to_event)
        step=st.session_state.replay_step
        for col,m in zip(st.columns(len(pair)),pair):
            with col,st.container(border=True):
                result=evaluation["methods"][m]["searches"][budget]["result"]
                current=min(step,len(result.trace))
                st.write(f"**{names[m]}**")
                selection=st.plotly_chart(tree_chart(result.nodes,result.trace,current,None,focus=True),width="stretch",theme=None,
                    key=f"tree_{m}",on_select="rerun",selection_mode="points")
                points=getattr(getattr(selection,"selection",None),"points",[]) or []
                if points:
                    custom=points[-1].get("customdata",[])
                    if custom: st.session_state[f"selected_tree_node_{m}"]=int(custom[-1])
                selected_node=st.session_state.get(f"selected_tree_node_{m}")
                if selected_node is not None:
                    node=next((n for n in result.nodes if n["id"]==selected_node),None)
                    selected_event=next((e for e in result.trace[:current] if e["node"]==selected_node),None)
                    if node is not None:
                        event_text="Queued" if selected_event is None else ("LB > BSF → PRUNE" if selected_event["action"] in {"prune","heuristic prune"} else selected_event["action"].upper())
                        st.caption(f'Selected · {node["label"]} · depth {node["depth"]} · {node.get("size",0):,} series · {node.get("leaf_pages",0):,} leaf pages · {event_text}')
                if current:
                    event=result.trace[current-1]
                    relation="LB > BSF" if event["action"] in {"prune","heuristic prune"} else event["action"].upper()
                    st.caption(f'{relation} · {event["label"]} · step {current}/{len(result.trace)}')
                    finished=current>=len(result.trace)
                    st.progress(current/max(len(result.trace),1),text=f'{"Completed" if finished else "Running"} · step {current} / {len(result.trace)}')
                    first_row=st.columns(2)
                    first_row[0].metric("Current node distance computations",f'{event.get("distance_computations",0):,}')
                    first_row[1].metric("Current node series pruned",f'{event.get("pruned_series",0):,}')
                    second_row=st.columns(2)
                    second_row[0].metric("Cumulative distance computations",f'{event.get("cumulative_distance_computations",0):,}')
                    second_row[1].metric("Cumulative series pruned",f'{event.get("cumulative_pruned_series",0):,}')
                    if finished:
                        final_row=st.columns(2)
                        final_row[0].metric("Leaf pages read",f'{result.metrics.get("leaves_read",0):,}')
                        final_row[1].metric("Pruned index nodes",f'{result.metrics.get("pruned_nodes",0):,}')
                else:
                    st.progress(0.,text=f"Ready · step 0 / {len(result.trace)}")
        if st.session_state.get("playing") and step<total:
            time.sleep(.4)
            st.session_state.replay_step=min(total,step+1)
            if st.session_state.replay_step>=total: st.session_state.playing=False
            st.rerun(scope="fragment")
    if st.button("Open full-screen exploration ↗",type="primary",width="stretch"):
        replay_dialog()
    payload=dict(dataset=name,query_id=exclude,query=query.tolist(),k=top_k,epsilon=epsilon,budget_ms=budget,timing="inference + traversal + candidate evaluation",ground_truth=evaluation["ground_truth"],
        methods={m:dict(coverage=evaluation["methods"][m]["coverage"],recall=evaluation["methods"][m]["searches"][budget]["matching"],metrics=evaluation["methods"][m]["searches"][budget]["result"].metrics) for m in selected})
    st.download_button("↓ Export single-query comparison",json.dumps(payload,indent=2),file_name="agenda-query.json",mime="application/json")

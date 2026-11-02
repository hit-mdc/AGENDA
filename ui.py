import numpy as np
import plotly.graph_objects as go

COLORS = dict(original="#8290A5", paa="#EF9E44", learned="#00A991", match="#4665D8")
METHOD_IDS = []
# Matplotlib tab20 palette, paired with distinct Plotly marker symbols.
_MPL_TAB20 = ["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a",
              "#d62728","#ff9896","#9467bd","#c5b0d5","#8c564b","#c49c94"]
_MARKERS = ["circle","square","diamond","triangle-up","triangle-down","x",
            "cross","pentagon","star","hexagon","hourglass","bowtie"]
METHOD_COLORS = dict(zip(METHOD_IDS,_MPL_TAB20))
METHOD_MARKERS = dict(zip(METHOD_IDS,_MARKERS))
_DASHES = ["solid","dash","dot","dashdot","longdash","longdashdot"]
METHOD_DASHES = {mid:_DASHES[i % len(_DASHES)] for i,mid in enumerate(METHOD_IDS)}


def register_styles(method_ids):
    for i,mid in enumerate(method_ids):
        METHOD_COLORS[mid]=_MPL_TAB20[i%len(_MPL_TAB20)]
        METHOD_MARKERS[mid]=_MARKERS[i%len(_MARKERS)]
        METHOD_DASHES[mid]=_DASHES[i%len(_DASHES)]


def chart_layout(fig, height=380):
    fig.update_layout(height=max(height,350), margin=dict(l=30,r=25,t=65,b=55),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif",color="#33445C",size=18),
        legend=dict(orientation="h", y=1.22,x=0,font=dict(size=18)), hovermode="x unified",
        hoverlabel=dict(font_size=18))
    fig.update_xaxes(showgrid=False,zeroline=False,automargin=True,tickfont=dict(size=18,color="#33445C"),title_font=dict(size=20,color="#33445C"))
    fig.update_yaxes(gridcolor="#E9EDF2",zeroline=False,automargin=True,tickfont=dict(size=18,color="#33445C"),title_font=dict(size=20,color="#33445C"))
    return fig


def series_chart(lines, height=300):
    fig = go.Figure()
    for i,line in enumerate(lines):
        name,values,color,step=line[:4]
        dash=line[4] if len(line)>4 else _DASHES[i % len(_DASHES)]
        width=3.2 if name=="Returned" else 2.4 if name in ["Matched ground truth","Missed ground truth"] else 2 if name!="Original" else 1.6
        fig.add_trace(go.Scatter(y=values, name=name, mode="lines",
            line=dict(color=color,width=width,dash=dash,shape="hv" if step else "linear")))
    fig.update_xaxes(title="Time point")
    fig.update_yaxes(title="Normalized amplitude")
    return chart_layout(fig,height)


def tree_chart(nodes, trace=None, step=0, prunable_roots=None, focus=None):
    """Render a focused search tree or, for diagnostics, the full hierarchy."""
    node_by_id={n["id"]:n for n in nodes}
    shown_trace=(trace or [])[:step]
    if focus is None:
        focus=trace is not None
    visible=set(node_by_id)
    if focus:
        visible={n['id'] for n in nodes if n['parent'] is None}
        def add_ancestors(nid):
            while nid is not None and nid in node_by_id:
                visible.add(nid)
                nid=node_by_id[nid]["parent"]
        for event in shown_trace:
            nid=event["node"]
            add_ancestors(nid)
            if event["action"]=="expand":
                visible.update(node_by_id[nid]["children"])
    display=[]
    for nid in sorted(visible):
        original=node_by_id[nid]
        display.append(dict(original,children=[c for c in original["children"] if c in visible]))
    pos = {}
    leaves = [n for n in display if not n["children"]]
    for i,n in enumerate(leaves): pos[n["id"]] = (i,-n["depth"])
    for n in reversed(display):
        if n["children"]: pos[n["id"]] = (np.mean([pos[c][0] for c in n["children"]]),-n["depth"])
    fig = go.Figure()
    edge_x,edge_y=[],[]
    for n in display:
        if n["parent"] is not None:
            p=pos[n["parent"]]; q=pos[n["id"]]
            edge_x.extend([p[0],q[0],None]); edge_y.extend([p[1],q[1],None])
    fig.add_trace(go.Scattergl(x=edge_x,y=edge_y,mode="lines",line=dict(color="#C9D4DF",width=1),hoverinfo="skip",showlegend=False))
    events={t["node"]:t for t in shown_trace}; status={nid:event["action"] for nid,event in events.items()}
    visit_order={}
    for order,event in enumerate(trace or [],1):
        visit_order.setdefault(event["node"],order)
    for event in shown_trace:
        if event["action"]=="expand":
            for child in node_by_id[event["node"]]["children"]:
                if child in visible and child not in status: status[child]="priority queue"
    colors={"expand":"#4665D8","prune":"#7C8796","heuristic prune":"#7C8796","scan":"#00A991","partial scan":"#E69A34","priority queue":"#F2B84B"}
    details=[]
    for n in display:
        if n.get("index_type")=="Dumpy":
            partition="Split dimensions: "+(", ".join(map(str,n.get("split_dimensions",[]))) or "leaf")
        else:
            cards=n.get("cardinalities",[])
            partition="Cardinalities: "+(" × ".join(map(str,cards[:4]))+(" …" if len(cards)>4 else ""))
        event=events.get(n["id"],{})
        lb_text="—" if not event else f'{event.get("lower_bound",float("nan")):.3f}'
        bsf_value=event.get("comparison_bsf",event.get("bsf"))
        bsf_text="—" if bsf_value is None else ("∞" if not np.isfinite(bsf_value) else f"{bsf_value:.3f}")
        order=visit_order.get(n["id"])
        order_label=f"Step #{order}" if order is not None else "Queued"
        details.append([order_label,n["label"],n["depth"],n.get("size",len(n["ids"])),status.get(n["id"],"unvisited"),partition,
                        n.get("leaf_pages",1),event.get("distance_computations",0),event.get("pruned_series",0),
                        event.get("cumulative_distance_computations",0),event.get("cumulative_pruned_series",0),
                        lb_text,bsf_text,n["id"]])
    marker_size=14 if focus else 8
    fig.add_trace(go.Scattergl(x=[pos[n["id"]][0] for n in display], y=[pos[n["id"]][1] for n in display],
        mode="markers",marker=dict(size=marker_size,color=[colors.get(status.get(n["id"]),"#E6EBF2") for n in display],line=dict(color="white",width=1)),
        customdata=details,
        hovertemplate="<b>%{customdata[0]} · %{customdata[1]}</b><br>LB: %{customdata[11]} · BSF: %{customdata[12]}<br>Depth %{customdata[2]}<br>%{customdata[3]:,} series · %{customdata[6]:,} leaf pages<br>%{customdata[5]}<br>Status: %{customdata[4]}<br>Distance computations here: %{customdata[7]:,}<br>Series pruned here: %{customdata[8]:,}<extra></extra>",showlegend=False))
    if shown_trace and shown_trace[-1]["node"] in pos:
        current_node=shown_trace[-1]["node"]
        fig.add_trace(go.Scattergl(x=[pos[current_node][0]],y=[pos[current_node][1]],mode="markers",
            marker=dict(size=28,color="rgba(0,0,0,0)",line=dict(color="#172B43",width=4)),
            customdata=[[current_node]],hoverinfo="skip",showlegend=False))
    for label,color in [("Expanded","#4665D8"),("Raw distances","#00A991"),("Pruned: LB > BSF","#7C8796"),("Priority queue","#F2B84B")]:
        fig.add_trace(go.Scattergl(x=[None],y=[None],mode="markers",name=label,marker=dict(size=11,color=color)))
    chart_layout(fig,520 if focus else 650)
    fig.update_layout(legend=dict(y=1.08,yanchor="bottom"),hovermode="closest",clickmode="event+select",dragmode="pan")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False,range=[-max(n["depth"] for n in display)-.4,.65])
    return fig


def index_overview_chart(nodes):
    depths=sorted({n["depth"] for n in nodes})
    internal=[sum(n["depth"]==d and bool(n["children"]) for n in nodes) for d in depths]
    leaves=[sum(n["depth"]==d and not n["children"] for n in nodes) for d in depths]
    fig=go.Figure()
    fig.add_bar(x=depths,y=internal,name="Internal nodes",marker_color="#4665D8")
    fig.add_bar(x=depths,y=leaves,name="Leaf pages",marker_color="#00A991")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Tree depth",dtick=1)
    fig.update_yaxes(title="Nodes",type="log")
    return chart_layout(fig,430)

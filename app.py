"""Configuration-driven interface for data series similarity search."""
import html
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backend.registry import read_config
from backend.external import ExternalStore,ExternalModelProvider,ExternalIndexProvider
from ui import COLORS,chart_layout,series_chart,tree_chart,METHOD_COLORS,METHOD_MARKERS,METHOD_DASHES,register_styles
from workspace import render_workspace,large_table

st.set_page_config(page_title="AGENDA · Time Series Search", page_icon="〰", layout="wide")
st.markdown('''<style>
html {font-size:18px}
.block-container {padding-top:4.5rem;max-width:1700px;padding-bottom:3rem}
[data-testid="stSidebar"] {border-right:1px solid #D8E2EA}
h1 {font-size:2.5rem!important;letter-spacing:-1px;line-height:1.2!important}
h2 {font-size:1.8rem!important} h3 {font-size:1.4rem!important}
[data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p {font-size:1.05rem;line-height:1.5}
[data-testid="stCaptionContainer"] p {font-size:.95rem!important;color:#516277}
[data-testid="stMetric"] {background:#FFF;border:1px solid #DCE4EC;padding:18px 22px;border-radius:12px}
[data-testid="stMetricValue"] {font-size:2.3rem!important}
[data-testid="stMetricLabel"] p {font-size:1rem!important}
.eyebrow {font-size:17px;font-weight:750;letter-spacing:2px;color:#008F7C;margin-bottom:12px}
.brand {font-size:32px;font-weight:800;letter-spacing:3px;color:#172B43}.brand span {color:#009D85}
.subtle {font-size:20px;color:#516277;line-height:1.5}
.badge {font-size:17px;color:#007C69;font-weight:650}
.large-table {overflow-x:auto;margin:12px 0 24px;border:1px solid #DCE4EC;border-radius:12px;background:#FFF}
.large-table table {border-collapse:collapse;width:100%;font-size:19px;text-align:left}
.large-table th {background:#EAF0F5;color:#263E52;font-weight:650;padding:15px 18px;text-align:left;white-space:nowrap}
.large-table td {padding:14px 18px;border-top:1px solid #E1E8EE;white-space:nowrap}
.large-table tr:nth-child(even) {background:#F8FAFC}
button[kind="primary"] {background:#008F7C;border-color:#008F7C;color:white}
div[data-testid="stDialog"] div[role="dialog"] {width:99vw;max-width:99vw;height:98vh;max-height:98vh;padding:10px 14px}
div[data-testid="stDialog"] div[role="dialog"] > div {max-width:none;width:100%}
div[data-testid="stDialog"] [data-testid="stMetric"] {padding:10px 14px}
div[data-testid="stDialog"] [data-testid="stMetricValue"] {font-size:1.75rem!important}
@media (min-width:1000px) {[data-testid="stSidebar"] {min-width:330px;max-width:370px}}
@media (max-width:800px) {html {font-size:17px}.subtle {font-size:18px}h1 {font-size:2rem!important}.block-container {padding:1.2rem}}
</style>''',unsafe_allow_html=True)


try:
    settings=read_config()
except FileNotFoundError:
    st.title("AGENDA")
    st.info("Create config.json with your datasets, DSA adapters, and index adapters, then run python prepare.py --config config.json.")
    st.stop()

@st.cache_resource
def providers(config):
    store=ExternalStore(config)
    models=ExternalModelProvider(store)
    return store,models,ExternalIndexProvider(store,models)

store,models,indexes=providers(settings)
catalog=models.list_models()
model_names={m['id']:m.get('name',m['id']) for m in catalog}
register_styles(list(model_names))
dataset_names={d['id']:d.get('name',d['id']) for d in settings['datasets']}
index_names={i['id']:i.get('name',i['id']) for i in settings['indexes']}
if 'page' not in st.session_state: st.session_state.page='Compare Methods'
if 'selected_methods' not in st.session_state: st.session_state.selected_methods=list(model_names)
st.session_state.selected_methods=[m for m in st.session_state.selected_methods if m in model_names]

with st.sidebar:
    st.markdown('<div class="brand"><span>≋</span> AGENDA</div>',unsafe_allow_html=True)
    st.caption('A General Deep Approximation Framework for Data Series Similarity Search')
    pages={'Compare Methods':'Explore DSA methods','Index Lab':'Explore D3S indexes','Query Set':'Query Set','Single Query':'Single Query'}
    page=st.radio('WORKSPACE',list(pages),format_func=pages.get,key='page')
    name=st.selectbox('Dataset',list(dataset_names),format_func=dataset_names.get)
    registered=settings.get('presets',{})
    presets={'All methods':list(model_names),**registered}
    def preset_changed():
        if st.session_state.preset in presets:
            st.session_state.selected_methods=[m for m in presets[st.session_state.preset] if m in model_names]
    st.selectbox('Comparison preset',['Custom']+list(presets),key='preset',on_change=preset_changed)
    selected=st.multiselect('Selected DSA methods',list(model_names),format_func=model_names.get,key='selected_methods')
    top_k=st.select_slider('Top-k neighbors',[1,5,10,20],value=10)
    index_type=st.selectbox('D3S index',list(index_names),format_func=index_names.get)
    epsilon=st.selectbox('Match tolerance ε',[1e-6,1e-5,1e-4],format_func=lambda v:f'{v:g}')
    cardinality=int(settings.get('cardinality',8))

try:
    data=store.data(name)
except (FileNotFoundError,ValueError) as exc:
    st.error(str(exc)); st.stop()
st.sidebar.caption(f'{len(data):,} series × {data.shape[1]}')
top_k=min(top_k,max(1,len(data)-1))
active=next((m for m in catalog if m['id'] in selected),catalog[0])
segments=int(active['dimensions'])
config=(name,segments,cardinality,index_type,top_k,epsilon)

def prepare_context(name,model_id,segments,cardinality,index_type):
    return indexes.prepare_context(name,model_id,index_type)

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


def approximation_view(query,method_id,key):
    ap=models.approximate(query,method_id,segments,cardinality,name)
    st.subheader('Raw series & reduced representation')
    st.plotly_chart(series_chart([('Raw series',query,COLORS['original'],False,'solid')],280),width='stretch',theme=None,key=f'signal_{key}')
    fig=go.Figure(go.Scatter(x=np.arange(1,len(ap.coefficients)+1),y=ap.coefficients,mode='lines+markers',name=model_names[method_id]))
    chart_layout(fig,300); fig.update_xaxes(title='Representation dimension'); fig.update_yaxes(title='Representation value')
    st.plotly_chart(fig,width='stretch',theme=None,key=f'approx_{key}')
    return ap

if page in ['Query Set','Single Query']:
    st.title('Compare methods across queries.' if page=='Query Set' else 'Compare methods on a single query.')
    def context_builder():
        return {mid:prepare_context(name,mid,segments,cardinality,index_type) for mid in selected}
    try:
        render_workspace(page,data,selected,models,indexes,context_builder,config,model_names)
    except (FileNotFoundError,RuntimeError,ValueError) as exc:
        st.error(str(exc))
elif page=='Compare Methods':
    st.title('Explore data series approximation methods')
    for group in dict.fromkeys(m['group'] for m in catalog):
        st.subheader(group)
        for m in [entry for entry in catalog if entry['group']==group]:
            with st.container(border=True):
                st.subheader(m.get('name',m['id'])); st.write(m['description'])
                st.caption(f"Output dimensions: {m['dimensions']}")
                def toggle(mid=m['id']):
                    values=list(st.session_state.selected_methods)
                    if mid in values: values.remove(mid)
                    else: values.append(mid)
                    st.session_state.selected_methods=values
                    st.session_state.preset='Custom'
                st.button('Remove from comparison' if m['id'] in selected else 'Add to comparison',key=f"method_{m['id']}",on_click=toggle)
    costs=[m for m in catalog if 'flops' in m and 'params' in m]
    if costs:
        st.subheader('Model efficiency')
        large_table(pd.DataFrame([{'Method':m.get('name',m['id']),'GFLOPs':m['flops'],'Parameters (M)':m['params']} for m in costs]))
        fig=go.Figure()
        for m in costs:
            fig.add_trace(go.Scatter(x=[m['flops']],y=[m['params']],name=m.get('name',m['id']),mode='markers+text',text=[m.get('name',m['id'])],marker=dict(color=METHOD_COLORS[m['id']])))
        chart_layout(fig,460); fig.update_xaxes(title='FLOPs (G)'); fig.update_yaxes(title='Parameters (M)')
        st.plotly_chart(fig,width='stretch',theme=None)
elif page=='Index Lab':
    st.title('Explore data series similarity search indexes')
    index_method=st.selectbox("Approximation method",list(model_names),format_func=model_names.get,key="index_method")
    index_context=prepare_context(name,index_method,segments,cardinality,index_type)
    nodes=index_context["nodes"]
    a,b,c,d=st.columns(4)
    a.metric("Dataset series",f"{len(data):,}"); b.metric("Total index nodes",len(nodes)); c.metric("Physical leaf pages",index_context["external_index"].get("leaf_pages",sum(not n["children"] for n in nodes))); d.metric("Tree depth",max((n["depth"] for n in nodes),default=0))
    left,right=st.columns([1.3,1])
    with left,st.container(border=True):
        st.subheader(f"{index_type} structure")
        st.plotly_chart(index_overview_chart(nodes),width="stretch",theme=None)
        with st.expander("Complete hierarchy · diagnostic"):
            st.plotly_chart(tree_chart(nodes,focus=False),width="stretch",theme=None)
    with right,st.container(border=True):
        st.subheader("Inspect a partition")
        nid=st.selectbox("Node",range(len(nodes)),format_func=lambda i:f'{nodes[i]["label"]} · {nodes[i].get("size",len(nodes[i].get("ids",[])))} series')
        node=nodes[nid]
        st.write(f'Depth **{node["depth"]}** · Type **{"Internal" if node["children"] else "Leaf"}**')
        st.plotly_chart(series_chart([(f'TS-{i:04d}',data[i],color,False) for i,color in zip(node["ids"][:3],["#00A991","#4665D8","#EF9E44"])],245),width="stretch",theme=None)
    with st.container(border=True):
        sid=st.slider("Series to encode",0,len(data)-1,min(42,len(data)-1)) if len(data)>1 else 0
        ap=approximation_view(data[sid],index_method,"index")
        st.subheader("Index symbols")
        st.code(" ".join(chr(97+int(s)) for s in ap.symbols),language=None)
        if len(ap.symbols)==len(ap.coefficients):
            large_table(pd.DataFrame({"Dimension":np.arange(1,len(ap.coefficients)+1),"Value":[f"{v:.3f}" for v in ap.coefficients],"Symbol":[chr(97+int(s)) for s in ap.symbols]}))
        else:
            large_table(pd.DataFrame({"Dimension":np.arange(1,len(ap.coefficients)+1),"Value":ap.coefficients}))

st.markdown('---')
st.caption('AGENDA: A General Deep Approximation Framework for Data Series Similarity Search')

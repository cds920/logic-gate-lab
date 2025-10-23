# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas.io.formats.style import Styler  # ★ pandas Styler 명시 import

# ─────────────────────────────────────────────────────────────────────────────
# Page config + Global style
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LogicLab: 게이트박스", page_icon="🔌", layout="wide")

PRIMARY   = "#4F46E5"   # indigo-600
SECONDARY = "#06B6D4"   # cyan-500
ACCENT    = "#22C55E"   # green-500
MUTED     = "#94A3B8"   # slate-400
INK       = "#24272e"   # dark ink

st.markdown(
    f"""
    <style>
      .block-container {{max-width: 1180px;}}

      /* Header (gradient) */
      .app-header {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {SECONDARY} 100%);
        padding: 18px 22px; border-radius: 20px; color: white;
        box-shadow: 0 10px 28px rgba(79,70,229,.25);
        margin-bottom: 16px;
      }}
      .app-header h1 {{margin: 0; font-size: 28px; font-weight: 800; letter-spacing:.2px}}
      .app-sub {{opacity:.9; font-size:14px; margin-top:4px}}

      /* Card */
      .card {{
        background: #ffffff; border-radius: 16px; padding: 18px 18px;
        box-shadow: 0 10px 24px rgba(15,23,42,.08); border: 1px solid #eef2ff;
        margin-top: 14px;
      }}

      /* Dataframe font */
      table.dataframe td, table.dataframe th {{font-size:14px}}

      /* ▼ 버튼 사이 간격 시각적 축소(패딩/폰트/최소폭 ↓) */
      .stButton>button {{
        border-radius: 10px; border: 1px solid #e2e8f0;
        padding: 4px 8px;
        font-size: 15px;
        min-width: 34px;
        line-height: 1.1;
        background: #ffffff;
        transition: all .15s ease;
      }}
      .stButton>button:hover {{transform: translateY(-1px); box-shadow:0 8px 18px rgba(2,6,23,.06)}}

      .stToggle label span {{font-weight:700}}

      /* Sidebar title gradient */
      section[data-testid="stSidebar"] .sidebar-title {{
        background: linear-gradient(90deg, {PRIMARY}, {SECONDARY});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="app-header">
      <h1>🔌 LogicLab: 게이트박스</h1>
      <div class="app-sub">게이트 도식과 타임라인으로 ‘바로 보이는’ 논리회로 실습</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────────────────────
def AND(a, b):  return int(a and b)
def OR(a, b):   return int(a or b)
def NOT(a):     return int(0 if a else 1)
def NAND(a, b): return NOT(AND(a, b))
def XOR(a, b):  return int((a and not b) or (not a and b))

GATE_FUNCS = {
    "NOT":  lambda a, b: NOT(a),
    "AND":  lambda a, b: AND(a, b),
    "OR":   lambda a, b: OR(a, b),
    "NAND": lambda a, b: NAND(a, b),
    "XOR":  lambda a, b: XOR(a, b),
}
GATES = ["NOT","AND","OR","NAND","XOR"]

BOOL_TEX = {
    "NOT":  r"Y=\overline{A}",
    "AND":  r"Y=A\cdot B",
    "OR":   r"Y=A+B",
    "NAND": r"Y=\overline{A\cdot B}",
    "XOR":  r"Y=\overline{A}\,B + A\,\overline{B}",
}

# pandas 버전 호환: index 숨기기 유틸
def _hide_index(styler: Styler) -> Styler:
    try:
        return styler.hide(axis="index")  # pandas >= 1.4
    except Exception:
        return styler.hide_index()        # pandas < 1.4

def truth_table(gate: str, a_sel=None, b_sel=None) -> Styler:
    if gate == "NOT":
        df = pd.DataFrame([{"A":0,"Y":GATE_FUNCS["NOT"](0,0)},
                           {"A":1,"Y":GATE_FUNCS["NOT"](1,0)}])
        return _hide_index(df.style)
    rows = [{"A":a,"B":b,"Y":GATE_FUNCS[gate](a,b)} for a in [0,1] for b in [0,1]]
    df = pd.DataFrame(rows)

    def _hl(row):
        if a_sel is not None and b_sel is not None and row["A"]==a_sel and row["B"]==b_sel:
            return ["background-color:#EEF2FF"]*len(row)
        return [""]*len(row)

    return _hide_index(df.style.apply(_hl, axis=1))

# ─────────────────────────────────────────────────────────────────────────────
# Gate drawing (Plotly) with LED glow
# ─────────────────────────────────────────────────────────────────────────────
INK = "#24272e"
ACCENT = "#22C55E"

def gate_figure(gate: str, A: int, B: int):
    line = INK
    lamp_on  = ACCENT
    lamp_off = "#9CA3AF"

    a_in, b_in = A, (B if gate!="NOT" else 0)
    Y = GATE_FUNCS[gate](a_in, b_in)

    fig = go.Figure()

    def in_port(cx, cy, label, val, draw=True):
        if draw:
            fig.add_shape(type="circle", x0=cx-0.5, x1=cx+0.5, y0=cy-0.5, y1=cy+0.5,
                          line=dict(color=line, width=4))
            fig.add_annotation(x=cx-0.85, y=cy, text=f"{label}={val}",
                               font=dict(size=14,color=INK),
                               showarrow=False, xanchor="right")

    in_port(1.4, 4.4, "A", a_in, True)
    in_port(1.4, 1.6, "B", b_in, gate!="NOT")

    wire_color = "#475569"
    fig.add_shape(type="line", x0=1.9, y0=4.4, x1=3.0, y1=4.4, line=dict(color=wire_color, width=4))
    if gate!="NOT":
        fig.add_shape(type="line", x0=1.9, y0=1.6, x1=3.0, y1=1.6, line=dict(color=wire_color, width=4))

    if gate in ["AND","NAND"]:
        fig.add_shape(type="rect", x0=3.0, y0=1.0, x1=5.0, y1=5.0, line=dict(color=line, width=4))
        fig.add_shape(type="path",
                      path="M 5.0 1.0 A 2.0 2.0 0 0 1 5.0 5.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(79,70,229,.04)")
    elif gate in ["OR","XOR"]:
        fig.add_shape(type="path",
                      path="M 3.0 1.0 Q 4.6 1.0 5.6 3.0 Q 4.6 5.0 3.0 5.0 Q 2.3 3.0 3.0 1.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(6,182,212,.05)")
        fig.add_shape(type="path",
                      path="M 3.0 1.0 Q 2.4 3.0 3.0 5.0",
                      line=dict(color=line, width=4))
        if gate == "XOR":
            fig.add_shape(type="path",
                          path="M 2.6 1.0 Q 2.0 3.0 2.6 5.0",
                          line=dict(color=line, width=3))
    else:  # NOT
        fig.add_shape(type="path",
            path="M 3.0 1.0 L 3.0 5.0 L 5.5 3.0 Z",
            line=dict(color=line, width=4), fillcolor="rgba(244,63,94,.05)")

    fig.add_annotation(x=4.2, y=3.0, text=gate,
                       font=dict(size=14,color=INK), showarrow=False)

    need_bubble = gate in ["NAND","NOT"]
    out_from = 5.7 if need_bubble else 5.3
    if need_bubble:
        fig.add_shape(type="circle", x0=5.45, x1=5.95, y0=2.75, y1=3.25,
                      line=dict(color=line, width=4), fillcolor="white")

    fig.add_shape(type="line", x0=out_from, y0=3.0, x1=7.2, y1=3.0, line=dict(color=wire_color, width=4))

    # LED glow
    led = lamp_on if Y==1 else lamp_off
    fig.add_shape(type="circle", x0=7.0, x1=8.4, y0=2.1, y1=3.9,
                  line=dict(color="rgba(0,0,0,0)", width=0),
                  fillcolor=("rgba(34,197,94,.20)" if Y==1 else "rgba(148,163,184,.15)"))
    fig.add_shape(type="circle", x0=7.2, x1=8.2, y0=2.3, y1=3.7, line=dict(color=led, width=6))
    fig.add_shape(type="circle", x0=7.28, x1=8.12, y0=2.38, y1=3.62, line=dict(color=led, width=0), fillcolor=led)
    fig.add_shape(type="circle", x0=7.36, x1=7.56, y0=3.28, y1=3.48, line=dict(color="white", width=0), fillcolor="white")
    fig.add_annotation(x=7.7, y=3.0, text=str(Y), font=dict(color="white", size=14), showarrow=False)

    fig.update_xaxes(range=[0,9], visible=False)
    fig.update_yaxes(range=[0,6], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=330, paper_bgcolor="rgba(0,0,0,0)")
    return fig, Y

# ─────────────────────────────────────────────────────────────────────────────
# Timeline plot + toggle row (alignment tuned)
# ─────────────────────────────────────────────────────────────────────────────
ALIGN_LEFT  = 0.160   # 그래프/토글 좌우 정렬 파라미터
ALIGN_RIGHT = 0.995
PAD_LEFT    = 0.160
PAD_RIGHT   = 0.005

def plot_track(values, label, n, color="#3B82F6"):
    fig = plt.figure(figsize=(7.2, 1.15))
    t = np.arange(n)
    plt.step(t, values, where="post", linewidth=2.2, color=color)
    plt.yticks([0,1], [0,1], fontsize=10)
    plt.ylim(-0.2,1.2)
    plt.ylabel(label, rotation=0, labelpad=20, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.xticks(t, fontsize=10)
    plt.subplots_adjust(left=ALIGN_LEFT, right=ALIGN_RIGHT, top=0.88, bottom=0.22)
    return fig

def render_toggle_row(seq, n, key_prefix, emoji_on="🔵", emoji_off="⚪",
                      left_pad=PAD_LEFT, right_pad=PAD_RIGHT):
    weights = [left_pad] + [1.0]*n + [right_pad]
    cols = st.columns(weights, gap="small")
    for i in range(n):
        with cols[i+1]:
            lab = emoji_on if seq[i]==1 else emoji_off
            if st.button(lab, key=f"{key_prefix}_{i}"):
                seq[i] = 1 - seq[i]
    return seq

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title"><h3>LogicLab: Menu</h3></div>', unsafe_allow_html=True)
page = st.sidebar.radio("페이지", ["스위치 실습(게이트→LED)","타임라인(최대 10칸)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# session
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0

# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Switch lab
# ─────────────────────────────────────────────────────────────────────────────
if page == "스위치 실습(게이트→LED)":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧪 스위치 실습 (Gate → LED)")
    gate = st.selectbox("Gate", GATES, index=1, key="lab_gate")

    left, mid, right = st.columns([0.58, 1.42, 0.9])

    with left:
        st.subheader("입력 스위치")
        st.session_state.A = 1 if st.toggle("A", value=bool(st.session_state.A), key="sw_A") else 0
        if gate == "NOT":
            B_val = 0
            st.toggle("B (NOT에서는 사용 안 함)", value=False, disabled=True)
        else:
            st.session_state.B = 1 if st.toggle("B", value=bool(st.session_state.B), key="sw_B") else 0
            B_val = st.session_state.B
        st.caption("스위치를 바꾸면 도면의 LED가 즉시 반응해요.")

    with mid:
        fig, Y = gate_figure(gate, st.session_state.A, B_val)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("현재 상태")
        c1, c2, c3 = st.columns(3)
        c1.metric("A", st.session_state.A)
        if gate != "NOT": c2.metric("B", B_val)
        c3.metric("Y", Y)

        st.subheader("부울대수")
        if gate == "XOR":
            st.latex(BOOL_TEX[gate])
            st.caption("동치표현:  $Y = A \\oplus B$")
        else:
            st.latex(BOOL_TEX[gate])

        st.subheader("Truth Table")
        a_sel = st.session_state.A
        b_sel = None if gate=="NOT" else B_val
        st.dataframe(truth_table(gate, a_sel, b_sel), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Timeline
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🕒 타임라인 — A/B/Y (최대 10칸)")
    gate = st.selectbox("Gate", GATES, index=4, key="tl_gate")  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 10, 10, step=1)

    if "A_seq" not in st.session_state or len(st.session_state.A_seq)!=n:
        st.session_state.A_seq = [0]*n
    if "B_seq" not in st.session_state or len(st.session_state.B_seq)!=n:
        st.session_state.B_seq = [0]*n

    # A: blue
    st.subheader("A")
    st.pyplot(plot_track(np.array(st.session_state.A_seq), "A", n, color="#3B82F6"), use_container_width=True)
    st.session_state.A_seq = render_toggle_row(st.session_state.A_seq, n, "tl_A", emoji_on="🔵", emoji_off="⚪")

    # B: orange
    st.subheader("B")
    st.pyplot(plot_track(np.array(st.session_state.B_seq), "B", n, color="#F59E0B"), use_container_width=True)
    st.session_state.B_seq = render_toggle_row(st.session_state.B_seq, n, "tl_B", emoji_on="🟠", emoji_off="⚪")

    # Y: green (computed)
    A_w = np.array(st.session_state.A_seq, dtype=int)
    B_w = np.array(st.session_state.B_seq, dtype=int)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a, b in zip(A_w, B_w)])

    st.subheader("Y")
    st.pyplot(plot_track(Y_w, "Y", n, color="#22C55E"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

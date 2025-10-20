# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="LogicLab: 게이트박스", page_icon="🔌", layout="wide")

# ── Logic ops ────────────────────────────────────────────────────────────────
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
    "NOT":  r"Y=\lnot A",
    "AND":  r"Y=A\cdot B",
    "OR":   r"Y=A+B",
    "NAND": r"Y=\lnot\!\left(A\cdot B\right)",
    "XOR":  r"Y=A\oplus B",
}

# ── Truth table ──────────────────────────────────────────────────────────────
def truth_table(gate: str) -> pd.DataFrame:
    if gate == "NOT":
        return pd.DataFrame([{"A":0,"Y":GATE_FUNCS["NOT"](0,0)},
                             {"A":1,"Y":GATE_FUNCS["NOT"](1,0)}])
    rows = [{"A":a,"B":b,"Y":GATE_FUNCS[gate](a,b)} for a in [0,1] for b in [0,1]]
    return pd.DataFrame(rows)

# ── Gate drawing (Plotly) ────────────────────────────────────────────────────
def gate_figure(gate: str, A: int, B: int):
    """
    좌표: x 0..9, y 0..6  (비율 고정)
    표준 느낌의 게이트 심볼 + LED 출력
    """
    line = "rgb(36,36,36)"
    lamp_on  = "rgb(34,197,94)"
    lamp_off = "rgb(120,120,120)"

    a_in, b_in = A, (B if gate!="NOT" else 0)
    Y = GATE_FUNCS[gate](a_in, b_in)

    fig = go.Figure()

    # 입력 포트(원 + 라벨)
    def in_port(cx, cy, label, val, draw=True):
        if draw:
            fig.add_shape(type="circle", x0=cx-0.5, x1=cx+0.5, y0=cy-0.5, y1=cy+0.5,
                          line=dict(color=line, width=4))
            fig.add_annotation(x=cx-0.8, y=cy, text=f"{label}={val}", showarrow=False, xanchor="right")

    in_port(1.4, 4.4, "A", a_in, True)
    in_port(1.4, 1.6, "B", b_in, gate!="NOT")

    # 배선(입력→게이트)
    fig.add_shape(type="line", x0=1.9, y0=4.4, x1=3.0, y1=4.4, line=dict(color=line, width=4))
    if gate!="NOT":
        fig.add_shape(type="line", x0=1.9, y0=1.6, x1=3.0, y1=1.6, line=dict(color=line, width=4))

    # 몸통
    if gate in ["AND","NAND"]:
        # ANSI AND: 직사각 + 반원
        fig.add_shape(type="rect", x0=3.0, y0=1.0, x1=5.0, y1=5.0, line=dict(color=line, width=4))
        # 반원(오른쪽)
        fig.add_shape(type="path",
                      path="M 5.0 1.0 A 2.0 2.0 0 0 1 5.0 5.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    elif gate in ["OR","XOR"]:
        # 뒤 곡선(OR 뒷선)
        fig.add_shape(type="path",
                      path="M 3.0 1.0 Q 4.6 1.0 5.6 3.0 Q 4.6 5.0 3.0 5.0 Q 2.3 3.0 3.0 1.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
        # 앞 곡선(입력쪽)
        fig.add_shape(type="path",
                      path="M 3.0 1.0 Q 2.4 3.0 3.0 5.0",
                      line=dict(color=line, width=4))
        # XOR은 추가 평행 곡선
        if gate == "XOR":
            fig.add_shape(type="path",
                          path="M 2.6 1.0 Q 2.0 3.0 2.6 5.0",
                          line=dict(color=line, width=3))
    else:  # NOT
        fig.add_shape(type="path",
            path="M 3.0 1.0 L 3.0 5.0 L 5.5 3.0 Z",
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")

    # 라벨(중앙)
    fig.add_annotation(x=4.2, y=3.0, text=gate, showarrow=False)

    # 인버터 버블(NAND/NOT)
    need_bubble = gate in ["NAND","NOT"]
    out_from = 5.7 if need_bubble else 5.3
    if need_bubble:
        fig.add_shape(type="circle", x0=5.45, x1=5.95, y0=2.75, y1=3.25, line=dict(color=line, width=4))

    # 출력선
    fig.add_shape(type="line", x0=out_from, y0=3.0, x1=7.2, y1=3.0, line=dict(color=line, width=4))

    # LED 아이콘
    led = lamp_on if Y==1 else lamp_off
    fig.add_shape(type="circle", x0=7.2, x1=8.2, y0=2.3, y1=3.7, line=dict(color=led, width=6))
    fig.add_shape(type="circle", x0=7.28, x1=8.12, y0=2.38, y1=3.62, line=dict(color=led, width=0), fillcolor=led)
    fig.add_shape(type="circle", x0=7.36, x1=7.56, y0=3.28, y1=3.48, line=dict(color="white", width=0), fillcolor="white")
    fig.add_annotation(x=7.7, y=3.0, text=str(Y), font=dict(color="white"), showarrow=False)

    # 축
    fig.update_xaxes(range=[0,9], visible=False)
    fig.update_yaxes(range=[0,6], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=380)

    return fig, Y

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["스위치 실습(게이트→LED)","타임라인(최대 10칸)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# 상태
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0

# ── Page 1: Switch Lab ───────────────────────────────────────────────────────
if page == "스위치 실습(게이트→LED)":
    st.header("🧪 스위치 실습 (Gate → LED)")
    gate = st.selectbox("Gate", GATES, index=1, key="lab_gate")

    left, mid, right = st.columns([0.55, 1.45, 0.9])

    with left:
        st.subheader("입력 스위치")
        st.session_state.A = 1 if st.toggle("A", value=bool(st.session_state.A), key="sw_A") else 0
        if gate == "NOT":
            B_val = 0
            st.toggle("B (NOT에서는 사용 안 함)", value=False, disabled=True)
        else:
            st.session_state.B = 1 if st.toggle("B", value=bool(st.session_state.B), key="sw_B") else 0
            B_val = st.session_state.B
        st.caption("스위치를 바꾸면 도면의 LED가 켜지거나 꺼집니다.")

    with mid:
        fig, Y = gate_figure(gate, st.session_state.A, B_val)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("현재 상태")
        st.metric("A", st.session_state.A)
        if gate != "NOT":
            st.metric("B", B_val)
        st.metric("Y", Y)

        st.subheader("Boolean Algebra")
        st.latex(BOOL_TEX[gate])

        st.subheader("Truth Table")
        st.dataframe(truth_table(gate), use_container_width=True, hide_index=True)

# ── Page 2: Timeline ─────────────────────────────────────────────────────────
else:
    st.header("🕒 타임라인 (칸을 클릭해 0/1 토글) — 최대 10칸")
    gate = st.selectbox("Gate", GATES, index=4, key="tl_gate")  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 10, 10, step=1)

    if "A_seq" not in st.session_state or len(st.session_state.A_seq)!=n:
        st.session_state.A_seq = [0]*n
    if "B_seq" not in st.session_state or len(st.session_state.B_seq)!=n:
        st.session_state.B_seq = [0]*n

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("랜덤 채우기"):
            st.session_state.A_seq = list(np.random.randint(0,2,n))
            st.session_state.B_seq = list(np.random.randint(0,2,n))
    with c2:
        if st.button("모두 0"):
            st.session_state.A_seq = [0]*n
            st.session_state.B_seq = [0]*n
    with c3:
        if st.button("모두 1(동상)"):
            st.session_state.A_seq = [1]*n
            st.session_state.B_seq = [1]*n

    st.markdown("#### A 행을 눌러 0/1 토글")
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.A_seq[i]==1 else "○"
        if c.button(lab, key=f"TA_{i}"):
            st.session_state.A_seq[i] = 1 - st.session_state.A_seq[i]

    st.markdown("#### B 행을 눌러 0/1 토글")
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.B_seq[i]==1 else "○"
        if c.button(lab, key=f"TB_{i}"):
            st.session_state.B_seq[i] = 1 - st.session_state.B_seq[i]

    A_w = np.array(st.session_state.A_seq, dtype=int)
    B_w = np.array(st.session_state.B_seq, dtype=int)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a, b in zip(A_w, B_w)])

    # 3트랙(A,B,Y), 0/1 눈금
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9,4.8), sharex=True)
    t = np.arange(n)
    axes[0].step(t, A_w, where="post"); axes[0].set_ylabel("A"); axes[0].set_yticks([0,1]); axes[0].set_ylim(-0.2,1.2)
    axes[1].step(t, B_w, where="post"); axes[1].set_ylabel("B"); axes[1].set_yticks([0,1]); axes[1].set_ylim(-0.2,1.2)
    axes[2].step(t, Y_w, where="post"); axes[2].set_ylabel("Y"); axes[2].set_yticks([0,1]); axes[2].set_ylim(-0.2,1.2)
    axes[2].set_xlabel(f"샘플(0..{n-1})")
    for ax in axes: ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(t)
    st.pyplot(fig, use_container_width=True)

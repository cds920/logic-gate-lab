# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="LogicLab: 게이트박스", page_icon="🔌", layout="wide")

# ── 논리 연산 ────────────────────────────────────────────────────────────────
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

# 부울대수(한글 표기, 오버라인 사용)
BOOL_TEX = {
    "NOT":  r"Y=\overline{A}",
    "AND":  r"Y=A\cdot B",
    "OR":   r"Y=A+B",
    "NAND": r"Y=\overline{A\cdot B}",
    "XOR":  r"Y=\overline{A}\,B + A\,\overline{B}",  # 동치: Y = A \oplus B
}

# ── 진리표 ───────────────────────────────────────────────────────────────────
def truth_table(gate: str) -> pd.DataFrame:
    if gate == "NOT":
        return pd.DataFrame([{"A":0,"Y":GATE_FUNCS["NOT"](0,0)},
                             {"A":1,"Y":GATE_FUNCS["NOT"](1,0)}])
    rows = [{"A":a,"B":b,"Y":GATE_FUNCS[gate](a,b)} for a in [0,1] for b in [0,1]]
    return pd.DataFrame(rows)

# ── 게이트 도면(표준 느낌) + LED ─────────────────────────────────────────────
def gate_figure(gate: str, A: int, B: int):
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
            fig.add_annotation(x=cx-0.85, y=cy, text=f"{label}={val}", showarrow=False, xanchor="right")

    in_port(1.4, 4.4, "A", a_in, True)
    in_port(1.4, 1.6, "B", b_in, gate!="NOT")

    # 배선
    fig.add_shape(type="line", x0=1.9, y0=4.4, x1=3.0, y1=4.4, line=dict(color=line, width=4))
    if gate!="NOT":
        fig.add_shape(type="line", x0=1.9, y0=1.6, x1=3.0, y1=1.6, line=dict(color=line, width=4))

    # 몸통
    if gate in ["AND","NAND"]:
        fig.add_shape(type="rect", x0=3.0, y0=1.0, x1=5.0, y1=5.0, line=dict(color=line, width=4))
        fig.add_shape(type="path",
                      path="M 5.0 1.0 A 2.0 2.0 0 0 1 5.0 5.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    elif gate in ["OR","XOR"]:
        fig.add_shape(type="path",
                      path="M 3.0 1.0 Q 4.6 1.0 5.6 3.0 Q 4.6 5.0 3.0 5.0 Q 2.3 3.0 3.0 1.0 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
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
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")

    # 라벨
    fig.add_annotation(x=4.2, y=3.0, text=gate, showarrow=False)

    # 인버터 버블
    need_bubble = gate in ["NAND","NOT"]
    out_from = 5.7 if need_bubble else 5.3
    if need_bubble:
        fig.add_shape(type="circle", x0=5.45, x1=5.95, y0=2.75, y1=3.25, line=dict(color=line, width=4))

    # 출력선
    fig.add_shape(type="line", x0=out_from, y0=3.0, x1=7.2, y1=3.0, line=dict(color=line, width=4))

    # LED
    led = lamp_on if Y==1 else lamp_off
    fig.add_shape(type="circle", x0=7.2, x1=8.2, y0=2.3, y1=3.7, line=dict(color=led, width=6))
    fig.add_shape(type="circle", x0=7.28, x1=8.12, y0=2.38, y1=3.62, line=dict(color=led, width=0), fillcolor=led)
    fig.add_shape(type="circle", x0=7.36, x1=7.56, y0=3.28, y1=3.48, line=dict(color="white", width=0), fillcolor="white")
    fig.add_annotation(x=7.7, y=3.0, text=str(Y), font=dict(color="white"), showarrow=False)

    fig.update_xaxes(range=[0,9], visible=False)
    fig.update_yaxes(range=[0,6], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=330)
    return fig, Y

# ── 파형 + 토글(정렬 맞춤) ───────────────────────────────────────────────────
def plot_track(values, label, n):
    fig = plt.figure(figsize=(7.2, 1.15))   # 작게
    t = np.arange(n)
    plt.step(t, values, where="post")
    plt.yticks([0,1], [0,1], fontsize=10)
    plt.ylim(-0.2,1.2)
    plt.ylabel(label, rotation=0, labelpad=20, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(t, fontsize=10)
    # 아래 토글과 수직 정렬 위해 좌우 여백 고정
    plt.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.22)
    return fig

def render_toggle_row(seq, n, key_prefix, left_pad=0.115, right_pad=0.015):
    # 그래프 좌/우 여백과 동일한 비율로 패딩 컬럼 추가
    weights = [left_pad] + [1.0]*n + [right_pad]
    cols = st.columns(weights, gap="small")
    for i in range(n):
        with cols[i+1]:
            lab = "●" if seq[i]==1 else "○"
            if st.button(lab, key=f"{key_prefix}_{i}"):
                seq[i] = 1 - seq[i]
    return seq

# ── 사이드바 ─────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["스위치 실습(게이트→LED)","타임라인(최대 10칸)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# 상태값
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0

# ── Page 1: 스위치 실습 ─────────────────────────────────────────────────────
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

        st.subheader("부울대수")
        if gate == "XOR":
            st.latex(BOOL_TEX[gate])
            st.caption("동치표현:  $Y = A \\oplus B$")
        else:
            st.latex(BOOL_TEX[gate])

        st.subheader("Truth Table")
        st.dataframe(truth_table(gate), use_container_width=True, hide_index=True)

# ── Page 2: 타임라인 ─────────────────────────────────────────────────────────
else:
    st.header("🕒 타임라인 — A/B/Y (최대 10칸)")
    gate = st.selectbox("Gate", GATES, index=4, key="tl_gate")  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 10, 10, step=1)

    # 시퀀스 초기화
    if "A_seq" not in st.session_state or len(st.session_state.A_seq)!=n:
        st.session_state.A_seq = [0]*n
    if "B_seq" not in st.session_state or len(st.session_state.B_seq)!=n:
        st.session_state.B_seq = [0]*n

    # A: 그래프 → 토글(수직 정렬)
    st.subheader("A")
    st.pyplot(plot_track(np.array(st.session_state.A_seq), "A", n), use_container_width=True)
    st.session_state.A_seq = render_toggle_row(st.session_state.A_seq, n, "tl_A")

    # B: 그래프 → 토글
    st.subheader("B")
    st.pyplot(plot_track(np.array(st.session_state.B_seq), "B", n), use_container_width=True)
    st.session_state.B_seq = render_toggle_row(st.session_state.B_seq, n, "tl_B")

    # Y: 계산 결과(토글 없음)
    A_w = np.array(st.session_state.A_seq, dtype=int)
    B_w = np.array(st.session_state.B_seq, dtype=int)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a, b in zip(A_w, B_w)])

    st.subheader("Y")
    st.pyplot(plot_track(Y_w, "Y", n), use_container_width=True)

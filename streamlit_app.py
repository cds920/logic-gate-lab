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
BASIC_GATES = ["NOT","AND","OR","NAND","XOR"]

BOOL_TEX = {
    "NOT":  r"Y=\lnot A",
    "AND":  r"Y=A\cdot B",
    "OR":   r"Y=A+B",
    "NAND": r"Y=\lnot\left(A\cdot B\right)",
    "XOR":  r"Y=A\oplus B",
}

# ── 진리표(게이트별) ─────────────────────────────────────────────────────────
def truth_table(gate_name: str) -> pd.DataFrame:
    if gate_name == "NOT":
        return pd.DataFrame([{"A":0,"Y":GATE_FUNCS["NOT"](0,0)},
                             {"A":1,"Y":GATE_FUNCS["NOT"](1,0)}])
    rows = []
    for a in [0,1]:
        for b in [0,1]:
            rows.append({"A":a,"B":b,"Y":GATE_FUNCS[gate_name](a,b)})
    return pd.DataFrame(rows)

# ── 게이트 도면(Plotly) + LED ────────────────────────────────────────────────
def gate_figure(gate:str, A:int, B:int):
    """
    좌표계: x 0..8, y 0..5
    입력 원(라벨), 게이트(실물 느낌), 우측 LED(아이콘 스타일)
    """
    lamp_on = "rgb(34,197,94)"
    lamp_off = "rgb(110,110,110)"
    line = "rgb(35,35,35)"

    # NOT은 B를 쓰지 않음
    a_in, b_in = A, (B if gate!="NOT" else 0)
    Y = GATE_FUNCS[gate](a_in, b_in)

    fig = go.Figure()

    # 입력 원 + 라벨
    fig.add_shape(type="circle", x0=0.9, x1=1.5, y0=3.5, y1=4.1, line=dict(color=line, width=4))
    fig.add_annotation(x=0.75, y=3.8, text=f"A={a_in}", showarrow=False, xanchor="right")
    if gate != "NOT":
        fig.add_shape(type="circle", x0=0.9, x1=1.5, y0=0.9, y1=1.5, line=dict(color=line, width=4))
        fig.add_annotation(x=0.75, y=1.2, text=f"B={b_in}", showarrow=False, xanchor="right")

    # 배선
    fig.add_shape(type="line", x0=1.5, y0=3.8, x1=2.6, y1=3.8, line=dict(color=line, width=4))
    if gate != "NOT":
        fig.add_shape(type="line", x0=1.5, y0=1.2, x1=2.6, y1=1.2, line=dict(color=line, width=4))

    # 게이트 몸통
    if gate in ["AND","NAND"]:
        fig.add_shape(type="rect", x0=2.6, y0=0.8, x1=4.3, y1=4.2, line=dict(color=line, width=4))
        fig.add_shape(type="path",  # 반원
                      path="M 4.3 0.8 A 1.7 1.7 0 0 1 4.3 4.2 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    elif gate in ["OR","XOR"]:
        if gate == "XOR":
            fig.add_shape(type="path",
                path="M 2.3 0.8 Q 1.9 2.5 2.3 4.2",
                line=dict(color=line, width=2))
        fig.add_shape(type="path",
            path="M 2.6 0.8 Q 3.7 0.8 4.5 2.0 Q 3.7 3.2 2.6 4.2 Q 2.1 2.5 2.6 0.8 Z",
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    else:  # NOT
        fig.add_shape(type="path",
            path="M 2.6 0.8 L 2.6 4.2 L 4.5 2.5 Z",
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")

    # 라벨
    fig.add_annotation(x=3.55, y=2.5, text=gate, showarrow=False)

    # 인버터 버블 (NAND/NOT)
    need_bubble = gate in ["NAND","NOT"]
    out_start_x = 4.7 if need_bubble else 4.5
    if need_bubble:
        fig.add_shape(type="circle", x0=4.52, x1=4.88, y0=2.32, y1=2.68, line=dict(color=line, width=4))

    # 출력선
    fig.add_shape(type="line", x0=out_start_x, y0=2.5, x1=6.0, y1=2.5, line=dict(color=line, width=4))

    # LED (아이콘)
    led_color = lamp_on if Y==1 else lamp_off
    fig.add_shape(type="circle", x0=6.0, x1=7.0, y0=1.8, y1=3.2, line=dict(color=led_color, width=6))
    fig.add_shape(type="circle", x0=6.08, x1=6.92, y0=1.88, y1=3.12, line=dict(color=led_color, width=0), fillcolor=led_color)
    fig.add_shape(type="circle", x0=6.16, x1=6.34, y0=2.86, y1=3.04, line=dict(color="white", width=0), fillcolor="white")
    fig.add_annotation(x=6.5, y=2.5, text=str(Y), font=dict(color="white"), showarrow=False)

    # 축/여백
    fig.update_xaxes(range=[0,8], visible=False)
    fig.update_yaxes(range=[0,5], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)

    return fig, Y

# ── 페이지 공통 UI ───────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["게이트 참고(그림·진리표·부울식)","스위치 실습(게이트→LED)","타임라인(최대 10칸)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# 세션값
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0

# ── 1) 게이트 참고 ───────────────────────────────────────────────────────────
if page == "게이트 참고(그림·진리표·부울식)":
    st.header("📘 게이트 참고 (그림 · 진리표 · 부울대수)")
    gate = st.selectbox("게이트 선택", BASIC_GATES, index=1)  # 기본 AND

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        # 고정 입력 예시(시각용) — AND는 (1,1), OR은 (1,0), NOT은 A=1
        A_ex = 1
        B_ex = 1 if gate not in ["NOT","OR","XOR","NAND"] else 0
        # 그림
        fig, Y = gate_figure(gate, A_ex, B_ex)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Boolean Algebra")
        st.latex(BOOL_TEX[gate])

    with c2:
        st.subheader("Truth Table")
        st.dataframe(truth_table(gate), use_container_width=True, hide_index=True)

# ── 2) 스위치 실습 ───────────────────────────────────────────────────────────
elif page == "스위치 실습(게이트→LED)":
    st.header("🧪 스위치 실습 (게이트 → LED)")
    gate = st.selectbox("게이트 선택", BASIC_GATES, index=1, key="lab_gate")

    left, mid, right = st.columns([0.6, 1.4, 0.8])

    with left:
        st.subheader("입력 스위치")
        st.session_state.A = 1 if st.toggle("A", value=bool(st.session_state.A), key="sw_A") else 0
        # NOT이면 B 비활성 안내
        if gate == "NOT":
            st.toggle("B (NOT에서는 사용 안 함)", value=False, disabled=True)
            b_val = 0
        else:
            st.session_state.B = 1 if st.toggle("B", value=bool(st.session_state.B), key="sw_B") else 0
            b_val = st.session_state.B
        st.caption("스위치를 바꾸면 도면의 LED가 켜지거나 꺼집니다.")

    with mid:
        fig, Y = gate_figure(gate, st.session_state.A, b_val)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("현재 상태")
        st.metric("A", st.session_state.A)
        if gate != "NOT":
            st.metric("B", b_val)
        st.metric("Y", Y)
        st.subheader("Truth Table")
        st.dataframe(truth_table(gate), use_container_width=True, hide_index=True)

# ── 3) 타임라인(최대 10칸) ───────────────────────────────────────────────────
else:
    st.header("🕒 타임라인 (칸을 클릭해 0/1 토글) — 최대 10칸")
    gate = st.selectbox("게이트 선택", BASIC_GATES, index=4, key="tl_gate")  # 기본 XOR
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

    # 3개 트랙(위에서부터 A/B/Y), 각 트랙 y축은 0/1로 표시
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9,4.8), sharex=True)
    t = np.arange(n)

    axes[0].step(t, A_w, where="post"); axes[0].set_ylabel("A"); axes[0].set_yticks([0,1]); axes[0].set_ylim(-0.2,1.2)
    axes[1].step(t, B_w, where="post"); axes[1].set_ylabel("B"); axes[1].set_yticks([0,1]); axes[1].set_ylim(-0.2,1.2)
    axes[2].step(t, Y_w, where="post"); axes[2].set_ylabel("Y"); axes[2].set_yticks([0,1]); axes[2].set_ylim(-0.2,1.2)
    axes[2].set_xlabel("샘플(0..{})".format(n-1))

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.xticks(t)
    st.pyplot(fig, use_container_width=True)

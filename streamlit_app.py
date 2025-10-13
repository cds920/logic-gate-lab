# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="LogicLab: 게이트박스", page_icon="🔌", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# 논리 연산
# ──────────────────────────────────────────────────────────────────────────────
def AND(a, b):  return int(a and b)
def OR(a, b):   return int(a or b)
def NOT(a):     return int(0 if a else 1)
def NAND(a, b): return NOT(AND(a, b))
def NOR(a, b):  return NOT(OR(a, b))
def XOR(a, b):  return int((a and not b) or (not a and b))
def XNOR(a, b): return NOT(XOR(a, b))

GATE_FUNCS = {
    "AND":    lambda a, b: AND(a, b),
    "OR":     lambda a, b: OR(a, b),
    "NAND":   lambda a, b: NAND(a, b),
    "NOR":    lambda a, b: NOR(a, b),
    "XOR":    lambda a, b: XOR(a, b),
    "XNOR":   lambda a, b: XNOR(a, b),
    "NOT(A)": lambda a, b: NOT(a),
    "NOT(B)": lambda a, b: NOT(b),
}
BASIC_GATES = ["AND","OR","NAND","NOR","XOR","XNOR","NOT(A)","NOT(B)"]

def truth_table(gate_name):
    rows = []
    for a in [0,1]:
        for b in [0,1]:
            rows.append({"A":a,"B":b, gate_name:GATE_FUNCS[gate_name](a,b)})
    return pd.DataFrame(rows)

def mark_current(df, a, b):
    df = df.copy()
    df.insert(0,"▶",["◻"]*len(df))
    df.loc[(df["A"]==a)&(df["B"]==b),"▶"]="▶"
    return df

def style_truth(df, a_sel, b_sel):
    def _hl(row):
        return ["background-color:#E6F4FF;" if (row["A"]==a_sel and row["B"]==b_sel) else "" ]*len(row)
    return df.style.apply(_hl, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 게이트 도면(Plotly) + 클릭 토글
# ──────────────────────────────────────────────────────────────────────────────
def gate_figure(gate:str, A:int, B:int):
    """
    x-range: 0..6, y-range: 0..4
    A 원 중심 (1,3), B 원 중심 (1,1)
    게이트 몸통: 간단 사각(2..4, 0.7..3.3) + 텍스트
    인버터 버블: 필요시 (4.15, 2) 반지름 0.12
    출력 램프: (5.5, 2) 반지름 0.45 (초록/회색)
    """
    lamp_on = "rgb(34,197,94)"   # 초록
    lamp_off = "rgb(90,90,90)"   # 회색
    line = "rgb(30,30,30)"

    # 출력 계산 (NOT 단일 입력 처리)
    a_in, b_in = (A, B)
    if gate == "NOT(A)":
        b_in = 0
    if gate == "NOT(B)":
        a_in = 0
    Y = GATE_FUNCS[gate](a_in, b_in)

    fig = go.Figure()

    # 클릭 타겟(투명 마커) — customdata로 'A', 'B' 태그
    fig.add_trace(go.Scatter(
        x=[1], y=[3], mode="markers+text",
        marker=dict(size=40, opacity=0.01),
        text=[f"A={A}"], textposition="middle left",
        customdata=["A"], hovertemplate="A 입력 원 클릭<extra></extra>",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[1], mode="markers+text",
        marker=dict(size=40, opacity=0.01),
        text=[f"B={B}"], textposition="middle left",
        customdata=["B"], hovertemplate="B 입력 원 클릭<extra></extra>",
        showlegend=False
    ))

    # 입력 원(시각용)
    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=1-0.35, x1=1+0.35, y0=3-0.35, y1=3+0.35,
                  line=dict(color=line, width=3))
    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=1-0.35, x1=1+0.35, y0=1-0.35, y1=1+0.35,
                  line=dict(color=line, width=3))

    # 배선(입력 → 게이트)
    fig.add_shape(type="line", x0=1+0.35, y0=3, x1=2, y1=3, line=dict(color=line, width=3))
    fig.add_shape(type="line", x0=1+0.35, y0=1, x1=2, y1=1, line=dict(color=line, width=3))

    # 게이트 몸통 (간단 직사각형) + 라벨
    fig.add_shape(type="rect", x0=2, y0=0.7, x1=4, y1=3.3, line=dict(color=line, width=3))
    fig.add_annotation(x=3, y=2, text=gate, showarrow=False)

    # 인버터 버블 필요 여부
    need_bubble = gate in ["NAND","NOR","XNOR"] or gate.startswith("NOT")
    out_start_x = 4.15 if need_bubble else 4

    if need_bubble:
        fig.add_shape(type="circle", x0=4.03, x1=4.27, y0=1.88, y1=2.12,
                      line=dict(color=line, width=3))

    # 출력선 + 출력 램프
    fig.add_shape(type="line", x0=out_start_x, y0=2, x1=5.05, y1=2, line=dict(color=line, width=3))
    # 램프 테두리
    lamp_color = lamp_on if Y==1 else lamp_off
    fig.add_shape(type="circle", x0=5.05, x1=5.95, y0=1.55, y1=2.45,
                  line=dict(color=lamp_color, width=6))
    fig.add_annotation(x=5.5, y=2, text=str(Y), font=dict(color=lamp_color), showarrow=False)

    # 축/여백 정리
    fig.update_xaxes(range=[0,6], visible=False)
    fig.update_yaxes(range=[0,4], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320)

    return fig, Y

# ──────────────────────────────────────────────────────────────────────────────
# 사이드바 / 페이지
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["게이트 뷰어","타임라인(클릭 편집)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# 상태
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0

# ──────────────────────────────────────────────────────────────────────────────
# 1) 게이트 뷰어 — 도면 자체 클릭으로 A/B 토글
# ──────────────────────────────────────────────────────────────────────────────
if page == "게이트 뷰어":
    st.header("🔎 게이트 뷰어 (도면 클릭 스위치)")

    gate = st.selectbox("게이트 선택", BASIC_GATES, index=0)
    fig, Y = gate_figure(gate, st.session_state.A, st.session_state.B)

    st.caption("도면의 A/B 원을 클릭하면 값이 토글됩니다.")
    clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="gateplot")

    # 클릭 처리
    if clicks:
        tag = clicks[0].get("customdata")
        if tag == "A":
            st.session_state.A = 1 - st.session_state.A
            st.experimental_rerun()
        if tag == "B":
            st.session_state.B = 1 - st.session_state.B
            st.experimental_rerun()

    # 우측 진리표(하이라이트)
    # NOT 단일 입력 처리용으로 보여줄 A/B는 계산에 쓴 값으로
    a_show, b_show = st.session_state.A, st.session_state.B
    if gate == "NOT(A)": b_show = 0
    if gate == "NOT(B)": a_show = 0
    df = truth_table(gate)
    dfm = mark_current(df, a_show, b_show)
    st.dataframe(style_truth(dfm, a_show, b_show), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) 타임라인 — 최대 12칸, x축 0,1,2,… 단위
# ──────────────────────────────────────────────────────────────────────────────
elif page == "타임라인(클릭 편집)":
    st.header("🕒 타임라인 (칸을 클릭해 0/1 토글)")

    gate = st.selectbox("게이트 선택", BASIC_GATES, index=4)  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 12, 12, step=1)

    # 시퀀스 상태
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

    fig = plt.figure(figsize=(9,3.2))
    t = np.arange(n)
    plt.step(t, A_w+2, where="post", label="A +2")
    plt.step(t, B_w+1, where="post", label="B +1")
    plt.step(t, Y_w+0, where="post", label=f"Y={gate}")
    plt.yticks([0,1,2,3], ["0","1","B","A"])
    plt.xticks(t)  # 0,1,2,… 1씩 증가
    plt.xlabel("샘플")
    plt.ylim(-0.5,3.5)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

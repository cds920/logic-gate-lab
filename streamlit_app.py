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
# 게이트 도면(Plotly) + 클릭 토글  — 실물 느낌 도형 & LED
# ──────────────────────────────────────────────────────────────────────────────
def gate_figure(gate:str, A:int, B:int):
    """
    좌표계: x 0..8, y 0..5
    A 중심 (1.2,3.8), B 중심 (1.2,1.2)
    몸통: AND D-모양 / OR·XOR 곡선 / NOT 삼각형
    버블: 필요 시 (4.9, 2.5) r=0.18
    LED: 테두리 + 채움 + 하이라이트(흰 점)
    """
    lamp_on = "rgb(34,197,94)"   # 초록
    lamp_off = "rgb(110,110,110)"# 회색
    line = "rgb(35,35,35)"

    # NOT 단일 입력 처리
    a_in, b_in = A, B
    if gate == "NOT(A)": b_in = 0
    if gate == "NOT(B)": a_in = 0
    Y = GATE_FUNCS[gate](a_in, b_in)

    fig = go.Figure()
    fig.update_layout(clickmode="event+select")  # 클릭 이벤트 확실히

    # ── 클릭 타겟(투명 마커) : A, B
    fig.add_trace(go.Scatter(
        name="A_target",
        x=[1.2], y=[3.8], mode="markers",
        marker=dict(size=80, color="rgba(0,0,0,0.001)"),
        customdata=["A"], hovertemplate="A 입력 클릭<extra></extra>", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        name="B_target",
        x=[1.2], y=[1.2], mode="markers",
        marker=dict(size=80, color="rgba(0,0,0,0.001)"),
        customdata=["B"], hovertemplate="B 입력 클릭<extra></extra>", showlegend=False
    ))

    # ── 입력 원 (시각)
    for cx, cy, label, val in [(1.2,3.8,"A",A),(1.2,1.2,"B",B)]:
        fig.add_shape(type="circle", x0=cx-0.45, x1=cx+0.45, y0=cy-0.45, y1=cy+0.45,
                      line=dict(color=line, width=4))
        fig.add_annotation(x=cx-0.85, y=cy, text=f"{label}={val}",
                           showarrow=False, xanchor="right")

    # ── 배선 (입력 → 게이트)
    fig.add_shape(type="line", x0=1.65, y0=3.8, x1=2.7, y1=3.8, line=dict(color=line, width=4))
    fig.add_shape(type="line", x0=1.65, y0=1.2, x1=2.7, y1=1.2, line=dict(color=line, width=4))

    # ── 게이트 몸통
    if gate in ["AND","NAND"]:
        # D 모양 (직사각 + 반원)
        fig.add_shape(type="rect", x0=2.7, y0=0.8, x1=4.4, y1=4.2, line=dict(color=line, width=4))
        # 반원
        fig.add_shape(type="path",
                      path="M 4.4 0.8 A 1.7 1.7 0 0 1 4.4 4.2 Z",
                      line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    elif gate in ["OR","NOR","XOR","XNOR"]:
        # OR 곡선
        # 앞쪽 얇은 곡선(XOR/XNOR)
        if gate in ["XOR","XNOR"]:
            fig.add_shape(type="path",
                path="M 2.4 0.8 Q 2.0 2.5 2.4 4.2",
                line=dict(color=line, width=2))
        fig.add_shape(type="path",
            path="M 2.7 0.8 Q 3.8 0.8 4.6 2.0 Q 3.8 3.2 2.7 4.2 Q 2.2 2.5 2.7 0.8 Z",
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")
    else:
        # NOT: 삼각형
        fig.add_shape(type="path",
            path="M 2.7 0.8 L 2.7 4.2 L 4.6 2.5 Z",
            line=dict(color=line, width=4), fillcolor="rgba(0,0,0,0)")

    # 라벨
    fig.add_annotation(x=3.6, y=2.5, text=gate, showarrow=False)

    # ── 인버터 버블
    need_bubble = gate in ["NAND","NOR","XNOR"] or gate.startswith("NOT")
    out_start_x = 4.8 if need_bubble else 4.6
    if need_bubble:
        fig.add_shape(type="circle", x0=4.62, x1=4.98, y0=2.32, y1=2.68, line=dict(color=line, width=4))

    # ── 출력선
    fig.add_shape(type="line", x0=out_start_x, y0=2.5, x1=6.0, y1=2.5, line=dict(color=line, width=4))

    # ── LED (테두리 + 내부 채움 + 하이라이트)
    led_color = lamp_on if Y==1 else lamp_off
    # 테두리
    fig.add_shape(type="circle", x0=6.0, x1=7.0, y0=1.8, y1=3.2, line=dict(color=led_color, width=6))
    # 내부 채움
    fig.add_shape(type="circle", x0=6.08, x1=6.92, y0=1.88, y1=3.12, line=dict(color=led_color, width=0), fillcolor=led_color)
    # 하이라이트(작은 흰 점)
    fig.add_shape(type="circle", x0=6.15, x1=6.35, y0=2.85, y1=3.05, line=dict(color="white", width=0), fillcolor="white")
    fig.add_annotation(x=6.5, y=2.5, text=str(Y), font=dict(color="white"), showarrow=False)

    # 축/여백
    fig.update_xaxes(range=[0,8], visible=False)
    fig.update_yaxes(range=[0,5], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=380)

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
    events = plotly_events(fig, click_event=True, select_event=False, hover_event=False, key="gateplot")

    # 클릭 처리 (trace 이름/ customdata 확인)
    if events:
        # streamlit-plotly-events는 선택된 포인트의 trace, pointIndex를 반환
        e = events[0]
        trace_name = e.get("curveNumber")  # trace index
        # 0: A_target, 1: B_target  (위에서 추가한 순서)
        if trace_name == 0:
            st.session_state.A = 1 - st.session_state.A
            st.experimental_rerun()
        elif trace_name == 1:
            st.session_state.B = 1 - st.session_state.B
            st.experimental_rerun()

    # NOT 단일 입력일 때 표시용 A/B
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

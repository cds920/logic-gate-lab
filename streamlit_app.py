# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from random import choice, randint

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
    """
    좌표: x 0..9, y 0..6  (비율 고정)
    표준 심볼 느낌의 게이트 + LED 출력
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
            fig.add_annotation(x=cx-0.85, y=cy, text=f"{label}={val}", showarrow=False, xanchor="right")

    in_port(1.4, 4.4, "A", a_in, True)
    in_port(1.4, 1.6, "B", b_in, gate!="NOT")

    # 배선(입력→게이트)
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

    # 인버터 버블(NAND/NOT)
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

    # 축
    fig.update_xaxes(range=[0,9], visible=False)
    fig.update_yaxes(range=[0,6], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360)

    return fig, Y

# ── 유틸: 파형 그리기 ────────────────────────────────────────────────────────
def plot_track(values, label, n):
    fig = plt.figure(figsize=(9,1.6))
    t = np.arange(n)
    plt.step(t, values, where="post")
    plt.yticks([0,1], [0,1])
    plt.ylim(-0.2,1.2)
    plt.ylabel(label)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(t)
    return fig

# ── 사이드바 ─────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["스위치 실습(게이트→LED)","타임라인(최대 10칸)","퀴즈(5문제)"])
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
elif page == "타임라인(최대 10칸)":
    st.header("🕒 타임라인 — A/B/Y (최대 10칸)")
    gate = st.selectbox("Gate", GATES, index=4, key="tl_gate")  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 10, 10, step=1)

    if "A_seq" not in st.session_state or len(st.session_state.A_seq)!=n:
        st.session_state.A_seq = [0]*n
    if "B_seq" not in st.session_state or len(st.session_state.B_seq)!=n:
        st.session_state.B_seq = [0]*n

    # A 트랙: 그래프 → 버튼(그래프 바로 아래, 같은 폭으로)
    st.subheader("A")
    st.pyplot(plot_track(np.array(st.session_state.A_seq), "A", n), use_container_width=True)
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.A_seq[i]==1 else "○"
        if c.button(lab, key=f"TA_{i}"):
            st.session_state.A_seq[i] = 1 - st.session_state.A_seq[i]

    # B 트랙
    st.subheader("B")
    st.pyplot(plot_track(np.array(st.session_state.B_seq), "B", n), use_container_width=True)
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.B_seq[i]==1 else "○"
        if c.button(lab, key=f"TB_{i}"):
            st.session_state.B_seq[i] = 1 - st.session_state.B_seq[i]

    # Y 트랙(계산 결과)
    A_w = np.array(st.session_state.A_seq, dtype=int)
    B_w = np.array(st.session_state.B_seq, dtype=int)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a, b in zip(A_w, B_w)])

    st.subheader("Y")
    st.pyplot(plot_track(Y_w, "Y", n), use_container_width=True)

# ── Page 3: 퀴즈(5문제) ──────────────────────────────────────────────────────
else:
    st.header("🧩 퀴즈 — 총 5문제")

    # 세션 상태
    if "quiz_qidx" not in st.session_state:
        st.session_state.quiz_qidx = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_data = {}

    def reset_quiz():
        st.session_state.quiz_qidx = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_data = {}

    colx, coly = st.columns([0.9,0.1])
    with coly:
        if st.button("다시 시작"):
            reset_quiz()

    qidx = st.session_state.quiz_qidx

    # ----- 문제 생성기 -----
    def gen_inputs(n=8):
        return [randint(0,1) for _ in range(n)]

    def draw_small_gate(g):
        # 시드 입력(보여주기용)
        fig, _ = gate_figure(g, 1, 1)
        return fig

    # 문제1: 게이트 그림 보고 이름 고르기(선다)
    def render_q1():
        gate = choice(GATES)
        st.session_state.quiz_data["A"] = gate
        st.subheader("Q1) 아래 **게이트 그림**은 무엇일까요?")
        st.plotly_chart(draw_small_gate(gate), use_container_width=True, config={"displayModeBar": False})
        ans = st.radio("정답 선택", GATES, key="q1_sel", horizontal=True)
        if st.button("정답 확인", key="q1_check"):
            if ans == gate:
                st.success("정답!")
                st.session_state.quiz_score += 1
            else:
                st.error(f"오답 😢  정답: {gate}")
            st.session_state.quiz_qidx += 1

    # 문제2: 진리표 보고 이름 고르기(선다)
    def render_q2():
        gate = choice(GATES)
        st.session_state.quiz_data["B"] = gate
        st.subheader("Q2) 아래 **진리표**의 게이트는 무엇일까요?")
        st.dataframe(truth_table(gate), use_container_width=True, hide_index=True)
        ans = st.radio("정답 선택", GATES, key="q2_sel", horizontal=True)
        if st.button("정답 확인", key="q2_check"):
            if ans == gate:
                st.success("정답!")
                st.session_state.quiz_score += 1
            else:
                st.error(f"오답 😢  정답: {gate}")
            st.session_state.quiz_qidx += 1

    # 문제3: 게이트와 입력값 보고 출력값이 1인지 O/X
    def render_q3():
        gate = choice(GATES)
        A = randint(0,1)
        B = 0 if gate=="NOT" else randint(0,1)
        Y = GATE_FUNCS[gate](A,B)
        st.subheader("Q3) **게이트와 입력값**이 주어졌을 때, 출력이 1인가요?")
        st.write(f"Gate: **{gate}**,  A={A}{'' if gate=='NOT' else f',  B={B}'}")
        ans = st.radio("출력이 1인가?", ["O","X"], key="q3_sel", horizontal=True)
        if st.button("정답 확인", key="q3_check"):
            ok = ("O" if Y==1 else "X")
            if ans == ok:
                st.success("정답!")
                st.session_state.quiz_score += 1
            else:
                st.error(f"오답 😢  정답: {ok} (출력={Y})")
            st.session_state.quiz_qidx += 1

    # 문제4: 입력 타임라인 → 출력 타임라인 그리기
    def render_q4():
        gate = choice(GATES)
        n = 8
        A = gen_inputs(n)
        B = [0]*n if gate=="NOT" else gen_inputs(n)
        Y = [GATE_FUNCS[gate](a,b) for a,b in zip(A,B)]
        st.subheader("Q4) **입력 타임라인**이 주어졌을 때, **출력 Y**를 직접 그려보세요.")
        st.write(f"Gate: **{gate}**")
        # 입력 그래프
        st.pyplot(plot_track(np.array(A), "A", n), use_container_width=True)
        if gate!="NOT":
            st.pyplot(plot_track(np.array(B), "B", n), use_container_width=True)
        # 답안 입력 버튼(Y)
        if "q4_ans" not in st.session_state or len(st.session_state.q4_ans)!=n:
            st.session_state.q4_ans = [0]*n
        st.markdown("#### Y를 눌러 0/1 토글")
        cols = st.columns(n, gap="small")
        for i,c in enumerate(cols):
            lab = "●" if st.session_state.q4_ans[i]==1 else "○"
            if c.button(lab, key=f"q4_{i}"):
                st.session_state.q4_ans[i] = 1 - st.session_state.q4_ans[i]
        # 제출
        if st.button("정답 확인", key="q4_check"):
            if st.session_state.q4_ans == Y:
                st.success("정답!")
                st.session_state.quiz_score += 1
            else:
                st.error("오답 😢  정답 파형을 아래에 보여줍니다.")
                st.pyplot(plot_track(np.array(Y), "정답 Y", n), use_container_width=True)
            st.session_state.quiz_qidx += 1

    # 문제5: 출력 타임라인만 보고 입력 타임라인 그리기
    # ※ 여러 해가 존재하는 문제를 피하려고 **NOT**으로만 출제(결정적 해 존재)
    def render_q5():
        gate = "NOT"
        n = 8
        A = gen_inputs(n)
        Y = [GATE_FUNCS[gate](a,0) for a in A]  # Y = NOT A
        st.subheader("Q5) **출력 타임라인(Y)** 만 보고, **입력 A**를 그리세요. (게이트: NOT)")
        st.pyplot(plot_track(np.array(Y), "Y", n), use_container_width=True)
        if "q5_ans" not in st.session_state or len(st.session_state.q5_ans)!=n:
            st.session_state.q5_ans = [0]*n
        st.markdown("#### A를 눌러 0/1 토글")
        cols = st.columns(n, gap="small")
        for i,c in enumerate(cols):
            lab = "●" if st.session_state.q5_ans[i]==1 else "○"
            if c.button(lab, key=f"q5_{i}"):
                st.session_state.q5_ans[i] = 1 - st.session_state.q5_ans[i]
        if st.button("정답 확인", key="q5_check"):
            if st.session_state.q5_ans == A:
                st.success("정답!")
                st.session_state.quiz_score += 1
            else:
                st.error("오답 😢  정답 파형을 아래에 보여줍니다.")
                st.pyplot(plot_track(np.array(A), "정답 A", n), use_container_width=True)
            st.session_state.quiz_qidx += 1

    # ----- 문제 진행 -----
    if qidx == 0:   render_q1()
    elif qidx == 1: render_q2()
    elif qidx == 2: render_q3()
    elif qidx == 3: render_q4()
    elif qidx == 4: render_q5()
    else:
        st.success(f"퀴즈 완료! 점수: **{st.session_state.quiz_score}/5**")
        if st.button("다시 시작하기"):
            reset_quiz()

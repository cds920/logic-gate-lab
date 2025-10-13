# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LogicLab: GateBox", page_icon="🔌", layout="wide")

# 상태 초기화
if "score" not in st.session_state:
    st.session_state.score = 0
if "q_index" not in st.session_state:
    st.session_state.q_index = 0
if "wrong" not in st.session_state:
    st.session_state.wrong = []

# ──────────────────────────────────────────────────────────────────────────────
# 게이트 정의
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
    "NOT(A)": lambda a, b: NOT(a),
    "NOT(B)": lambda a, b: NOT(b),
    "NAND":   lambda a, b: NAND(a, b),
    "NOR":    lambda a, b: NOR(a, b),
    "XOR":    lambda a, b: XOR(a, b),
    "XNOR":   lambda a, b: XNOR(a, b),
}
BASIC_GATES = ["AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT(A)", "NOT(B)"]

# 진리표 생성
def truth_table(gate_name):
    rows = []
    for a in [0, 1]:
        for b in [0, 1]:
            y = GATE_FUNCS[gate_name](a, b)
            rows.append({"A": a, "B": b, f"{gate_name}": y})
    return pd.DataFrame(rows)

# 현재 입력행 마킹(▶)
def mark_current(df, a, b):
    df = df.copy()
    df.insert(0, "▶", ["◻" for _ in range(len(df))])
    idx = (df["A"] == a) & (df["B"] == b)
    df.loc[idx, "▶"] = "▶"
    return df

# 스타일: 선택된 (A,B) 행 배경 하이라이트
def style_truth(df, a_sel, b_sel):
    def _hl(row):
        is_sel = (row["A"] == a_sel) and (row["B"] == b_sel)
        bg = "background-color: #E6F4FF;" if is_sel else ""
        return [bg] * len(row)
    return df.style.apply(_hl, axis=1)

# 게이트 다이어그램(Graphviz) 렌더
def render_gate_graph(gate_label, A_val, B_val, Y_val):
    y_color = "#22c55e" if Y_val == 1 else "#333333"
    dot = f"""
    digraph G {{
      rankdir=LR;
      node [shape=circle, fontsize=14, fontname="Arial"];

      A [label="A={A_val}"];
      B [label="B={B_val}"];

      gate [shape=box, style="rounded", label="{gate_label}", fontsize=16];

      A -> gate;
      B -> gate;

      Y [label="Y={Y_val}", color="{y_color}", fontcolor="{y_color}"];
      gate -> Y;

      {{rank=same; A B}}
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: GateBox")
page = st.sidebar.radio("페이지", ["게이트 뷰어", "타임라인", "2단 합성", "퀴즈", "대시보드"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# ──────────────────────────────────────────────────────────────────────────────
# 1) 게이트 뷰어  (A/B 스위치 = 진리표 바로 옆, 행 하이라이트 + 게이트 그림)
# ──────────────────────────────────────────────────────────────────────────────
if page == "게이트 뷰어":
    st.header("🔎 게이트 뷰어 (입력→출력 직관)")

    # 상단: 게이트 선택
    gate = st.selectbox("게이트 선택", BASIC_GATES, index=0, key="viewer_gate")

    # 본문 2열: 왼쪽(입출력/다이어그램), 오른쪽(스위치+진리표)
    left, right = st.columns([1, 1])

    with right:
        st.subheader("입력 스위치")
        r1, r2 = st.columns(2)
        with r1:
            A_local = st.toggle("A", value=False, key="viewer_A")
        with r2:
            B_local = st.toggle("B", value=False, key="viewer_B")
        A_i, B_i = int(A_local), int(B_local)

        st.subheader("진리표")
        df = truth_table(gate)
        df_mark = mark_current(df, A_i, B_i)  # ▶ 표시
        st.dataframe(
            style_truth(df_mark, A_i, B_i),   # 선택 행 하이라이트
            use_container_width=True,
            hide_index=True
        )

    with left:
        st.subheader("입/출력 패널")
        out = GATE_FUNCS[gate](A_i, B_i)
        st.write(f"**A:** `{A_i}`  |  **B:** `{B_i}`")
        led = "🟢 ON" if out == 1 else "⚫ OFF"
        st.metric(label=f"출력 {gate}", value=f"{out} ({led})")

        st.subheader("게이트 다이어그램")
        render_gate_graph(gate, A_i, B_i, out)

    st.info("오른쪽에서 A/B를 바꾸면, 진리표가 파란색으로 하이라이트되고 왼쪽 LED·다이어그램이 동시에 반응합니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 2) 타임라인 (파형 시각화)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "타임라인":
    st.header("🕒 타임라인 (사각파 → 출력 파형)")

    colL, colR = st.columns([1, 1])
    with colL:
        gate = st.selectbox("게이트 선택", BASIC_GATES, index=6, key="timeline_gate")
        n_cycles = st.slider("샘플 길이", 8, 64, 16, step=2)
        duty_A = st.slider("A 듀티(%)", 0, 100, 50, step=5)
        duty_B = st.slider("B 듀티(%)", 0, 100, 50, step=5)
        phase_B = st.slider("B 위상 지연(샘플)", 0, n_cycles-1, 0)
        st.caption("듀티는 1의 비율, 위상 지연은 B의 시작 위치를 밀어줍니다.")

    # 파형 생성
    def square_wave(n, duty, phase=0):
        arr = np.zeros(n, dtype=int)
        on_len = int(n * (duty/100))
        start = phase % n
        arr[start:start+on_len if start+on_len <= n else n] = 1
        if start+on_len > n:
            arr[:(start+on_len) % n] = 1
        return arr

    A_w = square_wave(n_cycles, duty_A, 0)
    B_w = square_wave(n_cycles, duty_B, phase_B)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a, b in zip(A_w, B_w)])

    with colR:
        fig = plt.figure(figsize=(7, 3))
        t = np.arange(n_cycles)
        plt.step(t, A_w+2, where="post", label="A +2")
        plt.step(t, B_w+1, where="post", label="B +1")
        plt.step(t, Y_w+0, where="post", label=f"Y={gate}")
        plt.yticks([0,1,2,3], ["0","1","B","A"])
        plt.xlabel("샘플")
        plt.ylim(-0.5, 3.5)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    st.success("XOR을 선택하고 B 위상을 살짝 밀어보세요. 두 입력이 다를 때만 출력이 1인 걸 파형으로 직관화할 수 있어요.")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 2단 합성 (간단 조합논리 빌더)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "2단 합성":
    st.header("🧱 2단 합성 (G1(A,B) ⊕ G2(A,B))")

    # 이 페이지는 자체 A/B 토글 사용 (뷰어 페이지와 독립)
    col_inputs = st.container()
    with col_inputs:
        i1, i2 = st.columns(2)
        with i1:
            A_local = st.toggle("A", value=False, key="compose_A")
        with i2:
            B_local = st.toggle("B", value=False, key="compose_B")
        A_i, B_i = int(A_local), int(B_local)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        g1 = st.selectbox("1단 게이트 G1", BASIC_GATES, index=0)
    with c2:
        comb = st.selectbox("결합 게이트", ["AND", "OR", "XOR", "XNOR", "NAND", "NOR"])
    with c3:
        g2 = st.selectbox("1단 게이트 G2", BASIC_GATES, index=1)

    G1 = GATE_FUNCS[g1](A_i, B_i)
    G2 = GATE_FUNCS[g2](A_i, B_i)
    Y  = GATE_FUNCS[comb](G1, G2)

    st.write(f"**입력** A={A_i}, B={B_i} →  **G1={g1}→{G1}**, **G2={g2}→{G2}**, **결합={comb}→Y={Y}**")
    st.metric("최종 출력 Y", Y)

    df_tt = pd.DataFrame(
        [{"A":a,"B":b,"G1":GATE_FUNCS[g1](a,b),"G2":GATE_FUNCS[g2](a,b),
          f"Y={comb}(G1,G2)":GATE_FUNCS[comb](GATE_FUNCS[g1](a,b), GATE_FUNCS[g2](a,b))}
         for a in [0,1] for b in [0,1]]
    )
    st.dataframe(mark_current(df_tt, A_i, B_i), use_container_width=True, hide_index=True)

    st.info("미션 예시: (A NAND B) OR (NOT A) 구성 후, 어떤 입력 조합에서 1이 되는지 찾아보세요.")

# ──────────────────────────────────────────────────────────────────────────────
# 4) 퀴즈
# ──────────────────────────────────────────────────────────────────────────────
elif page == "퀴즈":
    st.header("📝 개념 확인 퀴즈")

    questions = [
        {"type":"OX","q":"NAND만으로 모든 기본 게이트(AND/OR/NOT 등)를 만들 수 있다.","ans":"O","exp":"NAND는 기능적으로 완전(Functional completeness)합니다."},
        {"type":"MC","q":"다음과 동치인 게이트는?  ¬(A · B)","choices":["NOR","XOR","NAND","XNOR"],"ans":"NAND","exp":"¬(A·B)는 NAND의 정의와 동일합니다."},
        {"type":"MC","q":"XOR 출력이 1이 되는 경우는?","choices":["A=B","A≠B","항상 0","항상 1"],"ans":"A≠B","exp":"XOR은 두 입력이 다를 때 1입니다."},
        {"type":"MC","q":"NOR 게이트의 진리표에서 1이 되는 경우는?","choices":["A=0,B=0","A=1,B=0","A=0,B=1","A=1,B=1"],"ans":"A=0,B=0","exp":"NOR은 OR의 부정이므로 두 입력이 모두 0일 때만 1입니다."},
        {"type":"MC","q":"XNOR의 의미와 가장 가까운 것은?","choices":["동치","합","곱","부정"],"ans":"동치","exp":"XNOR은 A와 B가 같을 때 1 → 논리적 동치입니다."},
    ]

    q = questions[st.session_state.q_index % len(questions)]
    st.subheader(f"Q{st.session_state.q_index % len(questions) + 1}. {q['q']}")

    if q["type"] == "OX":
        sel = st.radio("선택", ["O","X"], horizontal=True)
    else:
        sel = st.radio("선택", q["choices"], index=0)

    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("제출"):
            if sel == q["ans"]:
                st.session_state.score += 1
                st.success("정답! ✅")
            else:
                st.session_state.wrong.append({"문항": q["q"], "선택": sel, "정답": q["ans"]})
                st.error("오답 ❌")
            st.info(q["exp"])
    with colB:
        if st.button("다음 문항"):
            st.session_state.q_index += 1
            st.experimental_rerun()

    st.divider()
    st.metric("누적 점수", st.session_state.score)
    if st.session_state.wrong:
        st.write("**오답노트**")
        st.dataframe(pd.DataFrame(st.session_state.wrong), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) 대시보드 (요약/가이드)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "대시보드":
    st.header("📊 수업 대시보드 & 사용 가이드")
    st.write("- 도입: **게이트 뷰어**에서 A/B 토글 → 출력 직관")
    st.write("- 개념: **타임라인**으로 XOR 등 파형 직관")
    st.write("- 실습: **2단 합성**으로 간단 조합논리 구성")
    st.write("- 확인: **퀴즈** 5~7문항, 오답노트 활용")
    st.info("발표는 시연 3분 + 설명 4분 구성. 제출물: Streamlit URL + 기획서/수업안 PDF")

# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="LogicLab: 게이트박스", page_icon="🔌", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# 기본 유틸
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

def truth_table(gate_name):
    rows = []
    for a in [0,1]:
        for b in [0,1]:
            y = GATE_FUNCS[gate_name](a, b)
            rows.append({"A": a, "B": b, f"{gate_name}": y})
    return pd.DataFrame(rows)

def mark_current(df, a, b):
    df = df.copy()
    df.insert(0, "▶", ["◻" for _ in range(len(df))])
    idx = (df["A"] == a) & (df["B"] == b)
    df.loc[idx, "▶"] = "▶"
    return df

def style_truth(df, a_sel, b_sel):
    def _hl(row):
        is_sel = (row["A"] == a_sel) and (row["B"] == b_sel)
        bg = "background-color: #E6F4FF;" if is_sel else ""
        return [bg] * len(row)
    return df.style.apply(_hl, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 게이트 SVG (ANSI 스타일 아이콘)  ← f-string 표현식 안에 역슬래시 없음
# ──────────────────────────────────────────────────────────────────────────────
def gate_svg(gate_label, A_val, B_val, Y_val, width=640, height=260):
    line = "#222"
    fill = "none"
    text = "#111"
    on_color = "#1f9d55" if Y_val == 1 else "#555"

    # 좌표
    left_x, mid_x, right_x = 90, 280, 520
    top_y, mid_y, bot_y = 70, 130, 190

    # 본체 path 데이터(문자열) 만들기
    pre_xor_path = ""     # XOR/XNOR 전면 얇은 곡선
    if gate_label in ["AND", "NAND"]:
        # D자 (좌 직선, 우 반원)
        body_path = f"M {mid_x-60},{top_y} L {mid_x-60},{bot_y} L {mid_x},{bot_y} A 60,60 0 0,0 {mid_x},{top_y} Z"
    elif gate_label in ["OR", "NOR", "XOR", "XNOR"]:
        if gate_label in ["XOR", "XNOR"]:
            pre_xor_path = f"M {mid_x-80},{top_y} C {mid_x-110},{mid_y} {mid_x-110},{mid_y} {mid_x-80},{bot_y}"
        body_path = (
            f"M {mid_x-70},{top_y} "
            f"C {mid_x-30},{top_y} {mid_x+30},{mid_y-40} {mid_x+30},{mid_y} "
            f"C {mid_x+30},{mid_y+40} {mid_x-30},{bot_y} {mid_x-70},{bot_y} "
            f"C {mid_x-40},{mid_y} {mid_x-40},{mid_y} {mid_x-70},{top_y} Z"
        )
    elif gate_label.startswith("NOT"):
        # 삼각형
        body_path = f"M {mid_x-60},{top_y} L {mid_x-60},{bot_y} L {mid_x+40},{mid_y} Z"
    else:
        body_path = f"M {mid_x-60},{top_y} L {mid_x-60},{bot_y} L {mid_x},{bot_y} A 60,60 0 0,0 {mid_x},{top_y} Z"

    need_bubble = gate_label in ["NAND", "NOR", "XNOR"] or gate_label.startswith("NOT")
    bubble_cx = mid_x + 44
    out_start = bubble_cx if need_bubble else mid_x + 30

    gate_text = (
        "그리고" if gate_label == "AND" else
        "또는"   if gate_label == "OR"  else
        "부정(A)" if gate_label == "NOT(A)" else
        "부정(B)" if gate_label == "NOT(B)" else
        gate_label
    )

    # SVG 조각(조건부 요소 미리 구성)
    pre_elem = f'<path d="{pre_xor_path}" stroke="{line}" fill="none" stroke-width="3"/>' if pre_xor_path else ""
    bubble_elem = f'<circle cx="{bubble_cx}" cy="{mid_y}" r="9" stroke="{line}" fill="#ffffff" stroke-width="3"/>' if need_bubble else ""

    svg = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
         xmlns="http://www.w3.org/2000/svg" style="background-color:white">
      <defs>
        <style>
          .t {{ font-family: 'DejaVu Sans', 'Arial', sans-serif; fill:{text}; font-size:18px; }}
        </style>
      </defs>

      <!-- 입력 원/라벨 -->
      <circle cx="{left_x}" cy="{top_y}" r="28" stroke="{line}" fill="none" stroke-width="3"/>
      <text x="{left_x-8}" y="{top_y+6}" class="t">A={A_val}</text>
      <circle cx="{left_x}" cy="{bot_y}" r="28" stroke="{line}" fill="none" stroke-width="3"/>
      <text x="{left_x-8}" y="{bot_y+6}" class="t">B={B_val}</text>

      <!-- 배선 -->
      <line x1="{left_x+28}" y1="{top_y}" x2="{mid_x-60}" y2="{top_y}" stroke="{line}" stroke-width="3"/>
      <line x1="{left_x+28}" y1="{bot_y}" x2="{mid_x-60}" y2="{bot_y}" stroke="{line}" stroke-width="3"/>

      <!-- 게이트 본체 -->
      {pre_elem}
      <path d="{body_path}" stroke="{line}" fill="{fill}" stroke-width="3"/>

      <!-- 게이트 라벨 -->
      <text x="{mid_x-22}" y="{mid_y+7}" class="t">{gate_text}</text>

      <!-- 버블(필요 시) -->
      {bubble_elem}

      <!-- 출력 -->
      <line x1="{out_start}" y1="{mid_y}" x2="{right_x-28}" y2="{mid_y}" stroke="{line}" stroke-width="3"/>
      <circle cx="{right_x}" cy="{mid_y}" r="40" stroke="{on_color}" fill="none" stroke-width="4"/>
      <text x="{right_x-14}" y="{mid_y+6}" class="t" fill="{on_color}">Y={Y_val}</text>
    </svg>
    """
    return svg

def show_gate_svg(gate_label, A_val, B_val, Y_val):
    st.markdown(gate_svg(gate_label, A_val, B_val, Y_val), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 페이지
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["게이트 뷰어", "타임라인(클릭 편집)", "2단 합성"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# ──────────────────────────────────────────────────────────────────────────────
# 1) 게이트 뷰어
# ──────────────────────────────────────────────────────────────────────────────
if page == "게이트 뷰어":
    st.header("🔎 게이트 뷰어 (입력→출력 직관)")

    gate = st.selectbox("게이트 선택", BASIC_GATES, index=0, key="viewer_gate")

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
        df_mark = mark_current(df, A_i, B_i)
        st.dataframe(style_truth(df_mark, A_i, B_i), use_container_width=True, hide_index=True)

    with left:
        st.subheader("입/출력 패널")
        out = GATE_FUNCS[gate](A_i, B_i)
        st.write(f"**A:** `{A_i}`  |  **B:** `{B_i}`")
        led = "🟢 켜짐" if out==1 else "⚫ 꺼짐"
        st.metric(label=f"출력 {gate}", value=f"{out} ({led})")

        st.subheader("게이트 다이어그램")
        show_gate_svg(gate, A_i, B_i, out)

    st.info("오른쪽에서 A/B를 바꾸면, 진리표 행이 파란색으로 하이라이트되고 왼쪽 LED·다이어그램이 동시에 반응합니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 2) 타임라인(클릭 편집)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "타임라인(클릭 편집)":
    st.header("🕒 타임라인 (칸을 클릭해 0/1 토글)")

    gate = st.selectbox("게이트 선택", BASIC_GATES, index=4, key="tl_gate")  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 8, 48, 16, step=2)

    # 세션 상태 준비
    if "A_seq" not in st.session_state or len(st.session_state.A_seq) != n:
        st.session_state.A_seq = [0]*n
    if "B_seq" not in st.session_state or len(st.session_state.B_seq) != n:
        st.session_state.B_seq = [0]*n

    c_btn1, c_btn2, c_btn3 = st.columns(3)
    with c_btn1:
        if st.button("랜덤 채우기"):
            st.session_state.A_seq = list(np.random.randint(0,2,n))
            st.session_state.B_seq = list(np.random.randint(0,2,n))
    with c_btn2:
        if st.button("모두 0"):
            st.session_state.A_seq = [0]*n
            st.session_state.B_seq = [0]*n
    with c_btn3:
        if st.button("모두 1(동상)"):
            st.session_state.A_seq = [1]*n
            st.session_state.B_seq = [1]*n

    st.markdown("#### A 행을 눌러 0/1 토글")
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.A_seq[i]==1 else "○"
        if c.button(lab, key=f"A_{i}"):
            st.session_state.A_seq[i] = 1 - st.session_state.A_seq[i]

    st.markdown("#### B 행을 눌러 0/1 토글")
    cols = st.columns(n, gap="small")
    for i, c in enumerate(cols):
        lab = "●" if st.session_state.B_seq[i]==1 else "○"
        if c.button(lab, key=f"B_{i}"):
            st.session_state.B_seq[i] = 1 - st.session_state.B_seq[i]

    A_w = np.array(st.session_state.A_seq, dtype=int)
    B_w = np.array(st.session_state.B_seq, dtype=int)
    Y_w = np.array([GATE_FUNCS[gate](int(a), int(b)) for a,b in zip(A_w,B_w)])

    fig = plt.figure(figsize=(9, 3.2))
    t = np.arange(n)
    plt.step(t, A_w+2, where="post", label="A +2")
    plt.step(t, B_w+1, where="post", label="B +1")
    plt.step(t, Y_w+0, where="post", label=f"Y={gate}")
    plt.yticks([0,1,2,3], ["0","1","B","A"])
    plt.xlabel("샘플")
    plt.ylim(-0.5, 3.5)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.success("팁: XOR을 선택한 뒤 A/B에서 서로 다른 칸을 몇 개 만들면, 두 입력이 다를 때만 출력이 1이 되는 게 바로 보입니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 2단 합성
# ──────────────────────────────────────────────────────────────────────────────
elif page == "2단 합성":
    st.header("🧱 2단 합성 (G1(A,B) → comb → G2(A,B))")

    i1, i2 = st.columns(2)
    with i1:
        A_local = st.toggle("A", value=False, key="compose_A")
    with i2:
        B_local = st.toggle("B", value=False, key="compose_B")
    A_i, B_i = int(A_local), int(B_local)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        g1 = st.selectbox("1단 게이트 G1", BASIC_GATES, index=0)
    with c2:
        comb = st.selectbox("결합 게이트", ["AND","OR","XOR","XNOR","NAND","NOR"])
    with c3:
        g2 = st.selectbox("1단 게이트 G2", BASIC_GATES, index=1)

    G1 = GATE_FUNCS[g1](A_i, B_i)
    G2 = GATE_FUNCS[g2](A_i, B_i)
    Y  = GATE_FUNCS[comb](G1, G2)

    st.write(f"**입력** A={A_i}, B={B_i} → **G1={g1}→{G1}**, **G2={g2}→{G2}**, **결합={comb}→Y={Y}**")
    st.metric("최종 출력 Y", Y)

    df_tt = pd.DataFrame(
        [{"A":a,"B":b,"G1":GATE_FUNCS[g1](a,b),"G2":GATE_FUNCS[g2](a,b),
          f"Y={comb}(G1,G2)":GATE_FUNCS[comb](GATE_FUNCS[g1](a,b), GATE_FUNCS[g2](a,b))}
         for a in [0,1] for b in [0,1]]
    )
    st.dataframe(mark_current(df_tt, A_i, B_i), use_container_width=True, hide_index=True)

    st.info("예시 미션: (A NAND B) OR (NOT A)을 만들어 보고, 어떤 입력 조합에서 1이 되는지 설명해 보세요.")

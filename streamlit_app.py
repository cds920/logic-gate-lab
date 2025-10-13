# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 클릭 좌표를 받기 위한 컴포넌트
from streamlit_drawable_canvas import st_canvas

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
# 클릭 가능한 게이트 다이어그램 (PIL로 그려서 Canvas에 표시)
#  - A/B 입력 원: 클릭하면 0/1 토글
#  - 출력 램프: 결과에 따라 회색/초록
#  - 게이트 모양: AND/OR/NOT 기본형 + 버블(NAND/NOR/XNOR)
# ──────────────────────────────────────────────────────────────────────────────
def draw_gate_image(gate, A, B, Y, W=900, H=330):
    img = Image.new("RGBA", (W, H), (255,255,255,0))
    d = ImageDraw.Draw(img)

    # 색상/굵기
    line = (34,34,34,255)
    lamp_on = (31,157,85,255)
    lamp_off = (120,120,120,255)
    thick = 6

    # 좌표
    Ax, Ay = 110, 95
    Bx, By = 110, 235
    r_in = 34
    gate_x = 330
    mid_y = (Ay+By)//2
    out_x = 760
    r_out = 46

    # 입력 원
    d.ellipse([Ax-r_in, Ay-r_in, Ax+r_in, Ay+r_in], outline=line, width=thick)
    d.ellipse([Bx-r_in, By-r_in, Bx+r_in, By+r_in], outline=line, width=thick)

    # 입력 라벨
    font = ImageFont.load_default()
    d.text((Ax-16, Ay-6), f"A={A}", fill=line, font=font)
    d.text((Bx-16, By-6), f"B={B}", fill=line, font=font)

    # 배선 (입력 → 게이트)
    d.line([Ax+r_in, Ay, gate_x-80, Ay], fill=line, width=thick)
    d.line([Bx+r_in, By, gate_x-80, By], fill=line, width=thick)

    # 게이트 본체
    # AND / NAND : D 모양
    if gate in ["AND","NAND"]:
        d.line([gate_x-80, Ay, gate_x-80, By], fill=line, width=thick)
        d.arc([gate_x-80, Ay-70, gate_x+60, By+70], start=270, end=90, fill=line, width=thick)
    # OR / NOR / XOR / XNOR : 타원형 근사
    elif gate in ["OR","NOR","XOR","XNOR"]:
        d.line([gate_x-90, mid_y, gate_x-60, mid_y], fill=line, width=thick)
        d.arc([gate_x-100, Ay-40, gate_x+70, By+40], start=300, end=60, fill=line, width=thick)
        d.arc([gate_x-80, Ay-70, gate_x+60, By+70], start=260, end=100, fill=line, width=thick)
        if gate in ["XOR","XNOR"]:
            # 앞쪽 얇은 곡선
            d.arc([gate_x-120, Ay-70, gate_x+20, By+70], start=260, end=100, fill=line, width=3)
    # NOT(A)/NOT(B) : 삼각형
    else:
        d.polygon([(gate_x-80, Ay-70), (gate_x-80, By+70), (gate_x+70, mid_y)], outline=line, width=thick)

    # 버블이 필요한 경우
    need_bubble = gate in ["NAND","NOR","XNOR"] or gate.startswith("NOT")
    bubble_cx, bubble_cy, bubble_r = gate_x+74, mid_y, 12
    if need_bubble:
        d.ellipse([bubble_cx-bubble_r, bubble_cy-bubble_r, bubble_cx+bubble_r, bubble_cy+bubble_r],
                  outline=line, fill=(255,255,255,255), width=thick)

    # 출력선
    start_x = bubble_cx + bubble_r if need_bubble else gate_x+56
    d.line([start_x, mid_y, out_x- r_out - 10, mid_y], fill=line, width=thick)

    # 출력 램프
    d.ellipse([out_x-r_out, mid_y-r_out, out_x+r_out, mid_y+r_out],
              outline=lamp_on if Y==1 else lamp_off, width=thick)
    d.text((out_x-12, mid_y-6), f"{Y}", fill=lamp_on if Y==1 else lamp_off, font=font)

    return img, (Ax,Ay,r_in), (Bx,By,r_in)

def point_in_circle(px, py, cx, cy, r):
    return (px-cx)**2 + (py-cy)**2 <= r**2

# ──────────────────────────────────────────────────────────────────────────────
# 사이드바 / 페이지
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("LogicLab: 게이트박스")
page = st.sidebar.radio("페이지", ["게이트 뷰어","타임라인(클릭 편집)"])
st.sidebar.caption("ⓘ 2학년 도제반 논리회로 도입/실습 확인용")

# 세션 상태
if "A" not in st.session_state: st.session_state.A = 0
if "B" not in st.session_state: st.session_state.B = 0
if "canvas_key" not in st.session_state: st.session_state.canvas_key = 0

# ──────────────────────────────────────────────────────────────────────────────
# 1) 게이트 뷰어 (도면 위 클릭 스위치)
# ──────────────────────────────────────────────────────────────────────────────
if page == "게이트 뷰어":
    st.header("🔎 게이트 뷰어 (도면 위 클릭 스위치)")

    gate = st.selectbox("게이트 선택", BASIC_GATES, index=0)

    # NOT 게이트는 단일 입력으로 동작하도록 처리
    A_in = st.session_state.A
    B_in = st.session_state.B
    if gate == "NOT(A)":
        B_in = 0
    elif gate == "NOT(B)":
        A_in = 0

    Y = GATE_FUNCS[gate](A_in, B_in)

    # 다이어그램 + 클릭 캔버스
    img, A_circle, B_circle = draw_gate_image(gate, A_in, B_in, Y)
    st.caption("도면의 A/B 원을 클릭하면 값이 토글됩니다.")
    canvas_res = st_canvas(
        background_image=img,
        width=img.width,
        height=img.height,
        drawing_mode="point",           # 클릭 좌표만 받기
        point_display_radius=1,
        stroke_width=0,
        update_streamlit=True,
        key=f"canvas_{st.session_state.canvas_key}",
        display_toolbar=False
    )

    # 클릭 처리
    if canvas_res.json_data and "objects" in canvas_res.json_data:
        objs = canvas_res.json_data["objects"]
        if len(objs) > 0:
            last = objs[-1]
            # 좌표는 좌상단 기준 (left, top)
            px, py = float(last.get("left",0)), float(last.get("top",0))
            (Ax,Ay,Ar) = A_circle
            (Bx,By,Br) = B_circle
            toggled = False
            if point_in_circle(px, py, Ax, Ay, Ar):
                st.session_state.A = 1 - st.session_state.A
                toggled = True
            elif point_in_circle(px, py, Bx, By, Br):
                st.session_state.B = 1 - st.session_state.B
                toggled = True
            if toggled:
                st.session_state.canvas_key += 1
                st.experimental_rerun()

    # 우측: 진리표 (현재 A/B와 동기)
    df = truth_table(gate)
    dfm = mark_current(df, A_in, B_in)
    st.dataframe(style_truth(dfm, A_in, B_in), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) 타임라인 (칸 클릭으로 0/1 토글, 최대 12칸, x축 1단위 표시)
# ──────────────────────────────────────────────────────────────────────────────
elif page == "타임라인(클릭 편집)":
    st.header("🕒 타임라인 (칸을 클릭해 0/1 토글)")
    gate = st.selectbox("게이트 선택", BASIC_GATES, index=4)  # 기본 XOR
    n = st.slider("샘플 길이(칸 수)", 4, 12, 12, step=1)

    # 세션 시퀀스 준비
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

    # 시각화 (x축 0..n-1, 1 간격)
    fig = plt.figure(figsize=(9, 3.2))
    t = np.arange(n)
    plt.step(t, A_w+2, where="post", label="A +2")
    plt.step(t, B_w+1, where="post", label="B +1")
    plt.step(t, Y_w+0, where="post", label=f"Y={gate}")
    plt.yticks([0,1,2,3], ["0","1","B","A"])
    plt.xticks(t)  # 0,1,2,… 1씩 증가
    plt.xlabel("샘플")
    plt.ylim(-0.5, 3.5)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.success("팁: XOR 선택 후 A/B에서 서로 다른 칸을 몇 개 만들면, 두 입력이 다를 때만 출력이 1이 되는 게 보입니다.")

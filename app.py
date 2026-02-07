# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes

st.set_page_config(page_title="超級校正台", page_icon="🛠️", layout="wide")

# ==========================================
# 🎛️ 側邊欄：超級控制面板
# ==========================================
st.sidebar.title("🛠️ 參數微調中心")
st.sidebar.write("請滑動滑桿，讓框框對準圓圈。")

# --- 1. 學號區設定 (獨立控制) ---
with st.sidebar.expander("🎓 1. 學號區 (藍框)", expanded=True):
    INFO_X_START = st.slider("X 起點 (左右)", 100, 300, 195, 1, help="學號區最左邊的開始位置")
    INFO_Y_ADJ   = st.slider("Y 微調 (上下)", -50, 50, 0, 1, help="學號區的垂直位置")
    INFO_GAP     = st.slider("格子間距", 50, 120, 90, 1, help="0到9之間的距離")

# --- 2. 作答區設定 (獨立控制) ---
with st.sidebar.expander("📝 2. 作答區 (綠框)", expanded=True):
    ANS_Y_ADJ = st.slider("Y 微調 (上下)", -50, 50, 0, 1)
    ANS_GAP   = st.slider("選項間距 (ABCD)", 50, 120, 95, 1, help="A和B之間的距離")
    
    st.write("--- 三欄位置微調 ---")
    L_OFFSET = st.slider("左欄 X 位置", 100, 300, 195, 1)
    M_OFFSET = st.slider("中欄 X 位置", 600, 800, 713, 1)
    R_OFFSET = st.slider("右欄 X 位置", 1100, 1350, 1247, 1)

# 固定參數 (方框大小)
INFO_BOX_SIZE = 35
ANS_BOX_SIZE = 34

# ==========================================
# 🎨 繪圖核心
# ==========================================
def draw_box(img, x, y, size, color, thickness=2):
    cv2.rectangle(img, (int(x), int(y)), (int(x+size), int(y+size)), color, thickness)

def visualize_calibration(image):
    # 1. 強制調整大小 (鎖定 A4 300dpi 規格)
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)

    # 2. 找定位點
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 1)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anchors = []
    debug_img = image.copy()
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
            # 🟡 畫定位點 (黃色)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 5) 

    anchors = sorted(anchors, key=lambda b: b[1])

    if len(anchors) < 25:
        return False, f"⚠️ 定位點不足 ({len(anchors)}/25)，請確認圖片是否清晰", debug_img

    # ==========================================
    # 🔵 畫學號區 (使用 INFO 參數)
    # ==========================================
    # 以前5個定位點為基準 (對應學號的5行)
    for i in range(5):
        if i >= len(anchors): break
        anchor = anchors[i]
        
        # 計算起始點：定位點X + 我們設定的X起點
        start_x = anchor[0] + INFO_X_START
        # 計算高度：定位點Y + 我們設定的Y微調
        start_y = anchor[1] + INFO_Y_ADJ
        
        # 畫 0-9
        for num in range(10):
            # 公式：起點 + (數字 * 間距)
            pos_x = start_x + (num * INFO_GAP)
            draw_box(debug_img, pos_x, start_y, INFO_BOX_SIZE, (255, 0, 0), 2) # 藍色

    # ==========================================
    # 🟢 畫作答區 (使用 ANS 參數)
    # ==========================================
    # 從第6個定位點開始 (index 5 ~ 24)
    for i in range(5, 25):
        if i >= len(anchors): break
        anchor = anchors[i]
        
        y_base = anchor[1] + ANS_Y_ADJ # 基準 Y
        x_base = anchor[0]             # 基準 X (定位點)

        # 左欄 (Q1-20)
        for j in range(4): # ABCD
            pos_x = x_base + L_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2) # 綠色

        # 中欄 (Q21-40)
        for j in range(4):
            pos_x = x_base + M_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2)

        # 右欄 (Q41-60)
        for j in range(4):
            pos_x = x_base + R_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2)

    return True, "繪製完成", debug_img

# ==========================================
# 🚀 主頁面
# ==========================================
st.title("🛠️ 閱卷參數校正台")
st.markdown("### 操作說明")
st.markdown("""
1. 上傳考卷。
2. 調整左側滑桿，直到：
   - **🔵 藍色框框** 對準上方的學號圈圈。
   - **🟢 綠色框框** 對準下方的答案圈圈。
3. **完成後，請把左側的所有數字截圖或複製給我。**
""")

uploaded_file = st.file_uploader("上傳 PDF 檔案", type="pdf")

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    # 取第一頁就好
    img = np.array(images[0])
    
    success, msg, res_img = visualize_calibration(img)
    
    if not success:
        st.error(msg)
    else:
        st.success("預覽圖已生成，請縮放檢視細節 👇")
    
    # 顯示圖片
    st.image(res_img, use_container_width=True, channels="BGR")
    
    # 顯示當前參數總結 (方便複製)
    st.divider()
    st.subheader("📋 目前參數 (請複製這段給我)")
    st.code(f"""
# 學生資訊區
INFO_X_START = {INFO_X_START}
INFO_GAP = {INFO_GAP}
INFO_Y_ADJ = {INFO_Y_ADJ}

# 作答區
ANS_GAP = {ANS_GAP}
L_OFFSET = {L_OFFSET}
M_OFFSET = {M_OFFSET}
R_OFFSET = {R_OFFSET}
ANS_Y_ADJ = {ANS_Y_ADJ}
    """)

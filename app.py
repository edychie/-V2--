# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes

st.set_page_config(page_title="無限校正台", page_icon="🔓", layout="wide")

st.title("🔓 閱卷參數校正台 (無限制版)")
st.warning("⚠️ 警告：此版本滑桿範圍極大 (0~2500)，請慢慢拖動以免框框飛出畫面。")

# ==========================================
# 🎛️ 側邊欄：參數設定
# ==========================================
st.sidebar.title("🎛️ 參數控制面板")

# --- 1. 學號區 (藍色) ---
st.sidebar.markdown("### 🔵 1. 學號區 (藍色)")
# 範圍加大到 0 ~ 2000
INFO_X_START = st.sidebar.slider("學號 X 起點", 0, 2000, 195, 1)
INFO_Y_ADJ   = st.sidebar.slider("學號 Y 上下微調", -300, 300, 0, 1)
INFO_GAP     = st.sidebar.slider("學號間距 (0-9)", 10, 300, 90, 1)

st.sidebar.markdown("---")

# --- 2. 作答區 (綠色) ---
st.sidebar.markdown("### 🟢 2. 作答區 (綠色)")
ANS_Y_ADJ = st.sidebar.slider("作答區 Y 上下微調", -300, 300, 0, 1)
ANS_GAP   = st.sidebar.slider("選項間距 (ABCD)", 10, 300, 95, 1)

st.sidebar.markdown("#### 三欄位置 (獨立設定)")
# 範圍全部開放 0 ~ 2500，你想把左欄放到右邊去都可以
L_OFFSET = st.sidebar.slider("左欄 (Q1-20) X位置", 0, 2500, 195, 1)
M_OFFSET = st.sidebar.slider("中欄 (Q21-40) X位置", 0, 2500, 713, 1)
R_OFFSET = st.sidebar.slider("右欄 (Q41-60) X位置", 0, 2500, 1247, 1)

# 固定參數
INFO_BOX_SIZE = 35
ANS_BOX_SIZE = 34

# ==========================================
# 🎨 繪圖核心
# ==========================================
def draw_box(img, x, y, size, color, thickness=2):
    # 防止畫出界導致報錯
    h, w = img.shape[:2]
    x, y = int(x), int(y)
    if x > 0 and y > 0 and x < w and y < h:
        cv2.rectangle(img, (x, y), (x+size, y+size), color, thickness)

def visualize_calibration(image):
    # 1. 強制調整大小
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
            # 🟡 定位點畫黃色
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 5) 

    anchors = sorted(anchors, key=lambda b: b[1])

    if len(anchors) < 25:
        return False, f"⚠️ 定位點不足 ({len(anchors)}/25)，請確認圖片清晰度或對比度", debug_img

    # ==========================================
    # 🔵 畫學號區 (前5個定位點)
    # ==========================================
    for i in range(5):
        if i >= len(anchors): break
        anchor = anchors[i]
        start_x = anchor[0] + INFO_X_START
        start_y = anchor[1] + INFO_Y_ADJ
        
        for num in range(10):
            pos_x = start_x + (num * INFO_GAP)
            draw_box(debug_img, pos_x, start_y, INFO_BOX_SIZE, (255, 0, 0), 2)

    # ==========================================
    # 🟢 畫作答區 (後20個定位點)
    # ==========================================
    for i in range(5, 25):
        if i >= len(anchors): break
        anchor = anchors[i]
        y_base = anchor[1] + ANS_Y_ADJ
        x_base = anchor[0]

        # 左欄
        for j in range(4):
            pos_x = x_base + L_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2)

        # 中欄
        for j in range(4):
            pos_x = x_base + M_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2)

        # 右欄
        for j in range(4):
            pos_x = x_base + R_OFFSET + (j * ANS_GAP)
            draw_box(debug_img, pos_x, y_base, ANS_BOX_SIZE, (0, 255, 0), 2)

    return True, "繪製完成", debug_img

# ==========================================
# 🚀 執行介面
# ==========================================
uploaded_file = st.file_uploader("上傳 PDF 檔案", type="pdf")

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    img = np.array(images[0])
    
    success, msg, res_img = visualize_calibration(img)
    
    if not success:
        st.error(msg)
    
    st.image(res_img, use_container_width=True, channels="BGR")
    
    st.divider()
    st.subheader("📋 調整完後，請複製這些數字給我：")
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

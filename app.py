# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes

# ==========================================
# 🖥️ 校正模式專用設定
# ==========================================
st.set_page_config(page_title="閱卷校正台", page_icon="🎛️", layout="wide")

st.sidebar.header("🎛️ 座標微調控制板")
st.sidebar.info("請調整下方滑桿，讓紅框對準圓圈。")

# --- 1. 全域位移 (控制所有框框) ---
st.sidebar.subheader("1. 全域位移 (整體移動)")
GLOBAL_X = st.sidebar.slider("↔️ X 左右微調", -100, 100, 0, help="正數往右，負數往左")
GLOBAL_Y = st.sidebar.slider("↕️ Y 上下微調", -100, 100, 0, help="正數往下，負數往上")

# --- 2. 題目區微調 (針對題目區) ---
st.sidebar.subheader("2. 題目區間距")
ANS_GAP_ADJ = st.sidebar.slider("📏 題目左右間距微調", -10, 10, 0)

# ==========================================
# 參數設定 (基礎值 + 微調值)
# ==========================================
# 基礎值 (來自 Colab)
BASE_INFO_X = 195
BASE_L_X = 195
BASE_M_X = 713
BASE_R_X = 1247
BASE_ANS_GAP = 95

# 應用微調
INFO_X_OFFSET = BASE_INFO_X + GLOBAL_X
L_OFFSET = BASE_L_X + GLOBAL_X
M_OFFSET = BASE_M_X + GLOBAL_X
R_OFFSET = BASE_R_X + GLOBAL_X
ANS_GAP = BASE_ANS_GAP + ANS_GAP_ADJ

INFO_GAP = 90
INFO_BOX_SIZE = 35
ANS_BOX_SIZE = 34
PIXEL_THRESHOLD = 200

# ==========================================
# 繪圖函式 (只畫圖，不計算)
# ==========================================
def draw_box(img, x, y, size, color=(0, 0, 255)):
    cv2.rectangle(img, (x, y), (x+size, y+size), color, 2)

def visualize_layout(image):
    # 強制 Resize
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)

    # 轉灰階找定位點
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 1)
    
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    
    debug_img = image.copy()
    
    # 找錨點
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 3) # 定位點畫黃色

    anchors = sorted(anchors, key=lambda b: b[1])

    if len(anchors) < 25:
        return False, f"⚠️ 定位點不足 ({len(anchors)}/25)，請確認圖片是否清晰", debug_img

    # 畫學號區
    if len(anchors) >= 5:
        # 為了示範，只畫第一行學號
        base_anchor = anchors[0]
        y_start = base_anchor[1] + GLOBAL_Y # 應用 Y 微調
        x_start = base_anchor[0] + INFO_X_OFFSET
        
        for i in range(10):
            draw_box(debug_img, x_start + (i * INFO_GAP), y_start, INFO_BOX_SIZE, (255, 0, 0)) # 藍色框

    # 畫作答區
    for i in range(5, 25):
        if i >= len(anchors): break
        anchor = anchors[i]
        y_a = anchor[1] + GLOBAL_Y # 應用 Y 微調
        x_a = anchor[0]

        # 左欄
        for j in range(4):
            draw_box(debug_img, x_a + L_OFFSET + (j * ANS_GAP), y_a, ANS_BOX_SIZE)
        # 中欄
        for j in range(4):
            draw_box(debug_img, x_a + M_OFFSET + (j * ANS_GAP), y_a, ANS_BOX_SIZE)
        # 右欄
        for j in range(4):
            draw_box(debug_img, x_a + R_OFFSET + (j * ANS_GAP), y_a, ANS_BOX_SIZE)

    return True, "繪製完成", debug_img

# ==========================================
# 主程式
# ==========================================
st.title("🎛️ 閱卷系統 - 視覺校正模式")
st.markdown("""
**說明：**
1. 上傳一份考卷。
2. 調整左側滑桿，直到 **紅色框框 (作答區)** 和 **藍色框框 (學號區)** 完美套在圓圈上。
3. **記下左側滑桿的數值**，並告訴 AI。
""")

uploaded_file = st.file_uploader("上傳考卷 PDF", type="pdf")

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    img = np.array(images[0])
    
    success, msg, result_img = visualize_layout(img)
    
    if not success:
        st.error(msg)
        st.image(result_img, caption="定位點偵測失敗示意圖", use_container_width=True)
    else:
        st.success(f"目前設定：X微調={GLOBAL_X}, Y微調={GLOBAL_Y}, 間距微調={ANS_GAP_ADJ}")
        st.image(result_img, caption="校正預覽圖 (請調整左側滑桿)", use_container_width=True, channels="BGR")
        st.divider()
        
        # 建立按鈕連結 (使用 st.link_button 最保險)
        st.link_button("📂 查看學生資料 (Google Sheet)", SHEET_URL, type="primary")

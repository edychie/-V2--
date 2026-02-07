# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes

# ==========================================
# ⚙️ 參數設定 (您提供的校正數據)
# ==========================================
GAS_URL = "https://script.google.com/macros/s/AKfycbxsvg7EjztbALAo47VDVR4v7vpzWunKnsvbv_ammmpfjfhX7_ZqdBPJxTWr56UhZr0u/exec"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1HEtNqxYTX0pZ3wEKh_G3AS0TSq2szhuF39ltFD73XEw/edit?usp=drive_link"

# 1. 學生資訊區 (藍色)
INFO_X_START = 282
INFO_GAP = 128
INFO_Y_ADJ = 12   # 往下 12
INFO_BOX_SIZE = 45 #稍微加大一點框框以確保包住

# 2. 作答區 (綠色)
ANS_Y_ADJ = 22    # 往下 22
ANS_GAP = 135     # 選項間距
ANS_BOX_SIZE = 45 # 稍微加大

# 三欄位置
L_OFFSET = 282
M_OFFSET = 1018
R_OFFSET = 1774

# 判定黑度的門檻 (如果發現有填滿卻沒讀到，可調低此值，例如 180)
PIXEL_THRESHOLD = 500

# ==========================================
# 🧠 核心邏輯
# ==========================================
def draw_debug_box(img, x, y, size, color):
    # 畫框框幫助除錯
    cv2.rectangle(img, (x, y), (x+size, y+size), color, 3)

def process_info_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    # 應用校正數據：定位點X + 起點X
    x_start = anchor[0] + offset
    # 應用校正數據：定位點Y + 微調Y
    y_start = anchor[1] + y_adj
    
    for i in range(10):
        x = x_start + (i * gap)
        # 確保不超出邊界
        if y_start < 0 or x < 0: continue
        
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)
        
        # 繪圖：有塗黑(>200)畫綠框，沒塗黑畫紅框
        color = (0, 255, 0) if score > PIXEL_THRESHOLD else (0, 0, 255)
        draw_debug_box(debug_img, x, y_start, box_s, color)
        
    return scores.index(max(scores))

def process_answer_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_a = anchor[0]
    # 應用校正數據：定位點Y + 微調Y
    y_a = anchor[1] + y_adj
    
    for i in range(4): # ABCD
        x = x_a + offset + (i * gap)
        
        # 防止越界
        if y_a < 0 or x < 0: 
            scores.append(0)
            continue

        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))
        
        # 繪圖
        color = (0, 255, 0) if scores[-1] > PIXEL_THRESHOLD else (0, 0, 255)
        draw_debug_box(debug_img, x, y_a, box_s, color)

    marked_indices = [idx for idx, s in enumerate(scores) if s > PIXEL_THRESHOLD]
    options = ['A', 'B', 'C', 'D']
    
    if len(marked_indices) == 0: return "X"   # 空白
    elif len(marked_indices) > 1: return "M"  # 複選(錯誤)
    else: return options[marked_indices[0]]

def analyze_paper(image):
    # ⭐ 關鍵：強制鎖定尺寸，確保你的校正數據有效
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)

    # 1. 轉灰階 & 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 1
    )
    
    debug_view = image.copy()
    
    # 2. 找定位點
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
            cv2.rectangle(debug_view, (x, y), (x+w, y+h), (0, 255, 255), 3)
    
    anchors = sorted(anchors, key=lambda b: b[1])
    
    if len(anchors) < 25:
        return False, f"定位點不足 (找到 {len(anchors)} 個，需要 25 個)", debug_view

    # 3. 解析內容
    try:
        # 傳入 INFO_Y_ADJ
        grade = process_info_row(thresh_inv, debug_view, anchors[0], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c1 = process_info_row(thresh_inv, debug_view, anchors[1], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c2 = process_info_row(thresh_inv, debug_view, anchors[2], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s1 = process_info_row(thresh_inv, debug_view, anchors[3], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s2 = process_info_row(thresh_inv, debug_view, anchors[4], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)

        result_data = {
            "grade": str(grade),
            "class": f"{c1}{c2}",
            "seat": f"{s1}{s2}",
            "answers": ""
        }
        
        ans_list = [""] * 60
        for i in range(5, 25):
            # 傳入 ANS_Y_ADJ
            # 左欄
            ans_list[i-5] = process_answer_row(thresh_inv, debug_view, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            # 中欄
            ans_list[i-5+20] = process_answer_row(thresh_inv, debug_view, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            # 右欄
            ans_list[i-5+40] = process_answer_row(thresh_inv, debug_view, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            
        result_data["answers"] = "".join(ans_list)
        return True, result_data, debug_view
        
    except Exception as e:
        return False, f"解析錯誤: {e}", debug_view

def upload_to_gas(data):
    if "script.google.com" not in GAS_URL: return True
    payload = {
        "grade": data["grade"], "className": data["class"],
        "seat": data["seat"], "answers": data["answers"]
    }
    try:
        r = requests.post(GAS_URL, json=payload, timeout=20)
        return r.status_code == 200
    except:
        return False

# ==========================================
# 🖥️ 網頁介面
# ==========================================
st.set_page_config(page_title="自動閱卷系統 (正式版)", page_icon="✅", layout="wide")

st.title("✅ 自動閱卷系統")
st.success(f"系統已校正：學號間距 {INFO_GAP} / 題目間距 {ANS_GAP}")

uploaded_files = st.file_uploader("選擇 PDF 檔案", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 開始閱卷"):
        st.divider()

        for idx, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"📄 {uploaded_file.name}")
            
            try:
                images = convert_from_bytes(uploaded_file.read())
                img = np.array(images[0])
                
                success, result, debug_img = analyze_paper(img)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if success:
                        st.success(f"辨識成功")
                        st.markdown(f"**學號：** `{result['grade']}年 {result['class']}班 {result['seat']}號`")
                        st.markdown("**答案預覽：**")
                        st.code(result['answers'], language="text")
                        
                        if upload_to_gas(result):
                            st.info("☁️ 資料已上傳至 Google Sheet")
                        else:
                            st.error("☁️ 上傳失敗 (請檢查網路或 GAS 連結)")
                    else:
                        st.error(f"❌ 辨識失敗：{result}")
                
                with col2:
                    st.caption("🔍 辨識結果確認 (紅框=未選, 綠框=已選)")
                    st.image(debug_img, use_container_width=True, channels="BGR")
                    
            except Exception as e:
                st.error(f"處理檔案時發生錯誤：{e}")

        st.divider()
        st.link_button("📂 開啟 Google Sheet 成績表", SHEET_URL, type="primary")






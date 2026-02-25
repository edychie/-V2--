# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes
import time

# ==========================================
# ⚙️ 參數設定 (保留您的校正數據)
# ==========================================
GAS_URL = "https://script.google.com/macros/s/AKfycbxsvg7EjztbALAo47VDVR4v7vpzWunKnsvbv_ammmpfjfhX7_ZqdBPJxTWr56UhZr0u/exec"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1HEtNqxYTX0pZ3wEKh_G3AS0TSq2szhuF39ltFD73XEw/edit?usp=drive_link"

# 1. 學生資訊區 (藍色)
INFO_X_START = 282
INFO_GAP = 128
INFO_Y_ADJ = 12   
INFO_BOX_SIZE = 45 

# 2. 作答區 (綠色)
ANS_Y_ADJ = 22    
ANS_GAP = 135     
ANS_BOX_SIZE = 45 

# 三欄位置
L_OFFSET = 282
M_OFFSET = 1018
R_OFFSET = 1774

# 判定黑度的門檻
PIXEL_THRESHOLD = 550

# ==========================================
# 🧠 核心邏輯 (保持不變)
# ==========================================
def process_info_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1] + y_adj
    
    for i in range(10):
        x = x_start + (i * gap)
        if y_start < 0 or x < 0: continue
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)
        
    return scores.index(max(scores))

def process_answer_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_a = anchor[0]
    y_a = anchor[1] + y_adj
    
    for i in range(4): # ABCD
        x = x_a + offset + (i * gap)
        if y_a < 0 or x < 0: 
            scores.append(0)
            continue
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))

    marked_indices = [idx for idx, s in enumerate(scores) if s > PIXEL_THRESHOLD]
    options = ['A', 'B', 'C', 'D']
    
    # === 🛑 修正區塊開始 ===
    if len(marked_indices) == 0: 
        return "" # 沒作答回傳空字串 (不要回傳 X，這樣 GAS 算分才不會出錯)
    else: 
        # 將所有超過門檻的選項組合成字串，例如 [0, 3] 會變成 "AD"
        return "".join([options[i] for i in marked_indices])
    # === 🛑 修正區塊結束 ===

def analyze_paper_simple(image):
    # 強制鎖定尺寸
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)

    # 轉灰階 & 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 1
    )
    
    # 找定位點
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
    
    anchors = sorted(anchors, key=lambda b: b[1])
    
    if len(anchors) < 25:
        return False, f"定位點不足 (找到 {len(anchors)} 個)"

    # 解析內容
    try:
        grade = process_info_row(thresh_inv, anchors[0], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c1 = process_info_row(thresh_inv, anchors[1], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c2 = process_info_row(thresh_inv, anchors[2], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s1 = process_info_row(thresh_inv, anchors[3], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s2 = process_info_row(thresh_inv, anchors[4], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)

        # ... (前面的程式碼不變)
        result_data = {
            "grade": str(grade),
            "class": f"{c1}{c2}",
            "seat": f"{s1}{s2}",
            "answers": [] # 先預設為空陣列
        }
        
        ans_list = [""] * 60
        for i in range(5, 25):
            ans_list[i-5] = process_answer_row(thresh_inv, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            ans_list[i-5+20] = process_answer_row(thresh_inv, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            ans_list[i-5+40] = process_answer_row(thresh_inv, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            
        # === 🛑 修正區塊開始 ===
        # 不要用 "".join(ans_list)，直接把整個陣列 (List) 傳給 GAS
        # 這樣 requests.post 發送 JSON 時，GAS 就會收到一個乾淨的陣列
        result_data["answers"] = ans_list 
        return True, result_data
        # === 🛑 修正區塊結束 ===
        
    except Exception as e:
        return False, f"解析錯誤: {e}"
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
# 🖥️ 網頁介面 (極簡版)
# ==========================================
st.set_page_config(page_title="自動閱卷系統", page_icon="📝", layout="centered")

st.title("自動閱卷系統")
st.subheader("歡迎使用，本網站適用於列定位點的特定答案卡，請注意")

uploaded_files = st.file_uploader("請選擇 PDF 檔案 (可多選)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 開始閱卷", type="primary"):
        st.divider()
        
        # 進度條容器
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_container = st.container()
        
        total_files = len(uploaded_files)
        success_count = 0
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # 更新進度
            current_progress = (idx + 1) / total_files
            progress_bar.progress(current_progress)
            status_text.text(f"⏳ 正在處理 ({idx+1}/{total_files}): {uploaded_file.name} ...")
            
            try:
                images = convert_from_bytes(uploaded_file.read())
                img = np.array(images[0])
                
                # 執行分析 (不回傳圖片，只回傳數據)
                success, result = analyze_paper_simple(img)
                
                with result_container:
                    if success:
                        # 嘗試上傳
                        if upload_to_gas(result):
                            st.success(f"✅ {uploaded_file.name} - 辨識成功且已上傳 (學號: {result['grade']}-{result['class']}-{result['seat']})")
                            success_count += 1
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} - 辨識成功但上傳失敗")
                    else:
                        st.error(f"❌ {uploaded_file.name} - 失敗: {result}")
                        
            except Exception as e:
                with result_container:
                    st.error(f"❌ {uploaded_file.name} - 發生錯誤: {e}")

        # 完成後顯示
        status_text.text(f"🏁 處理完成！ 成功: {success_count} / 總共: {total_files}")
        
        st.divider()
        # Google Sheet 按鈕
        st.link_button("📂 開啟 Google Sheet 成績表", SHEET_URL, type="primary", use_container_width=True)


# ﻿-*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import requests
from pdf2image import convert_from_bytes
import time

# ==========================================
# ⚙️ 參數設定 (您的核心數據)
# ==========================================
GAS_URL = "https://script.google.com/macros/s/AKfycbxsvg7EjztbALAo47VDVR4v7vpzWunKnsvbv_ammmpfjfhX7_ZqdBPJxTWr56UhZr0u/exec"

# 學生資訊區
INFO_X_OFFSET = 195
INFO_GAP = 90
INFO_BOX_SIZE = 35

# 三欄式題目區
ANS_GAP = 95
ANS_BOX_SIZE = 34
L_OFFSET = 195   
M_OFFSET = 713   
R_OFFSET = 1247  

# 判定黑度的門檻
PIXEL_THRESHOLD = 200

# ==========================================
# 🧠 核心邏輯
# ==========================================
def process_info_row(thresh_img, anchor, offset, gap, box_s):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1]
    for i in range(10):
        x = x_start + (i * gap)
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))
    
    max_val = max(scores)
    return scores.index(max_val)

def process_answer_row(thresh_img, anchor, offset, gap, box_s):
    scores = []
    x_a, y_a, _, _ = anchor
    for i in range(4):
        x = x_a + offset + (i * gap)
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))

    marked_indices = [idx for idx, s in enumerate(scores) if s > PIXEL_THRESHOLD]
    options = ['A', 'B', 'C', 'D']
    
    if len(marked_indices) == 0: return "X"
    elif len(marked_indices) > 1: return "M"
    else: return options[marked_indices[0]]

def analyze_paper_stream(image):
    # 1. 轉灰階 & 自適應二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 1
    )
    
    # 2. 找定位點
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
    
    anchors = sorted(anchors, key=lambda b: b[1])
    
    if len(anchors) < 25:
        return False, "定位點不足", None

    # 3. 解析內容
    grade = process_info_row(thresh_inv, anchors[0], INFO_X_OFFSET, INFO_GAP, INFO_BOX_SIZE)
    c1 = process_info_row(thresh_inv, anchors[1], INFO_X_OFFSET, INFO_GAP, INFO_BOX_SIZE)
    c2 = process_info_row(thresh_inv, anchors[2], INFO_X_OFFSET, INFO_GAP, INFO_BOX_SIZE)
    s1 = process_info_row(thresh_inv, anchors[3], INFO_X_OFFSET, INFO_GAP, INFO_BOX_SIZE)
    s2 = process_info_row(thresh_inv, anchors[4], INFO_X_OFFSET, INFO_GAP, INFO_BOX_SIZE)

    result_data = {
        "grade": str(grade),
        "class": f"{c1}{c2}",
        "seat": f"{s1}{s2}",
        "answers": ""
    }
    
    ans_list = [""] * 60
    for i in range(5, 25):
        # 左中右三欄
        ans_list[i-5] = process_answer_row(thresh_inv, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        ans_list[i-5+20] = process_answer_row(thresh_inv, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        ans_list[i-5+40] = process_answer_row(thresh_inv, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        
    result_data["answers"] = "".join(ans_list)
    
    # 回傳一張標記過的圖給使用者看
    debug_img = image.copy()
    cv2.putText(debug_img, f"ID: {grade}-{c1}{c2}-{s1}{s2}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    return True, result_data, debug_img

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
# 🖥️ 網頁介面 (UI)
# ==========================================
st.set_page_config(page_title="自動閱卷系統", page_icon="📝")

st.title("📝 自動閱卷小幫手")
st.markdown("請直接將掃描好的 **PDF 考卷** 拖曳到下方，系統會自動辨識並上傳成績。")

# 檔案上傳區
uploaded_files = st.file_uploader("選擇 PDF 檔案 (可多選)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 開始閱卷"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        fail_count = 0
        total_files = len(uploaded_files)

        st.divider()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"正在處理：{uploaded_file.name} ...")
            
            try:
                # 記憶體內直接轉檔，不存硬碟
                images = convert_from_bytes(uploaded_file.read())
                img = np.array(images[0])
                
                success, result, debug_img = analyze_paper_stream(img)
                
                if success:
                    # 顯示結果
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.success(f"✅ {uploaded_file.name}")
                        st.write(f"**{result['grade']}年 {result['class']}班 {result['seat']}號**")
                        
                        # 上傳 GAS
                        if upload_to_gas(result):
                            st.caption("☁️ 成績已上傳")
                        else:
                            st.error("☁️ 上傳失敗")
                            
                    with col2:
                        # 顯示縮圖
                        st.image(debug_img, caption="辨識結果", use_container_width=True)
                    
                    success_count += 1
                else:
                    st.error(f"❌ {uploaded_file.name} 辨識失敗：{result}")
                    fail_count += 1
                    
            except Exception as e:
                st.error(f"❌ {uploaded_file.name} 發生錯誤：{e}")
                fail_count += 1
            
            # 更新進度條
            progress_bar.progress((idx + 1) / total_files)
            time.sleep(0.5)

        status_text.text("處理完成！")

        st.success(f"📊 結算：成功 {success_count} 份 / 失敗 {fail_count} 份")


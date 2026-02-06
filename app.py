# -*- coding: utf-8 -*-
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
SHEET_URL = "https://docs.google.com/spreadsheets/d/1HEtNqxYTX0pZ3wEKh_G3AS0TSq2szhuF39ltFD73XEw/edit?usp=drive_link"

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
# 🧠 核心邏輯 (輕量化版 - 不產圖)
# ==========================================
def process_info_row(thresh_img, anchor, offset, gap, box_s):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1]
    for i in range(10):
        x = x_start + (i * gap)
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))
    return scores.index(max(scores))

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

def analyze_paper_stream_lite(image):
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
        return False, "定位點不足 (少於25個)"

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
        ans_list[i-5] = process_answer_row(thresh_inv, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        ans_list[i-5+20] = process_answer_row(thresh_inv, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        ans_list[i-5+40] = process_answer_row(thresh_inv, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE)
        
    result_data["answers"] = "".join(ans_list)
    return True, result_data

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
# 🖥️ 網頁介面 (簡潔版)
# ==========================================
st.set_page_config(page_title="自動閱卷系統", page_icon="📝")

st.title("📝 自動閱卷小幫手")
st.markdown("請直接將掃描好的 **PDF 考卷** 拖曳到下方，系統會自動辨識並上傳成績。記得只能傳列定位點的圖喔!")

# 檔案上傳區
uploaded_files = st.file_uploader("選擇 PDF 檔案 (可多選)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 開始閱卷"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        fail_count = 0
        total_files = len(uploaded_files)

        # 建立一個容器來顯示即時日誌
        log_container = st.container()

        st.divider()

        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"⏳ 正在處理 ({idx+1}/{total_files})：{uploaded_file.name} ...")
            
            try:
                # 轉檔
                images = convert_from_bytes(uploaded_file.read())
                img = np.array(images[0])
                
                # 辨識
                success, result = analyze_paper_stream_lite(img)
                
                if success:
                    # 上傳 GAS
                    upload_success = upload_to_gas(result)
                    
                    # 顯示簡短成功訊息
                    with log_container:
                        msg = f"✅ **{uploaded_file.name}** | 學號：{result['grade']}年{result['class']}班{result['seat']}號"
                        if upload_success:
                            st.success(f"{msg} (☁️ 已上傳)")
                        else:
                            st.warning(f"{msg} (☁️ 上傳失敗)")
                    
                    success_count += 1
                else:
                    with log_container:
                        st.error(f"❌ **{uploaded_file.name}** 辨識失敗：{result}")
                    fail_count += 1
                    
            except Exception as e:
                with log_container:
                    st.error(f"❌ **{uploaded_file.name}** 發生錯誤：{e}")
                fail_count += 1
            
            # 更新進度條
            progress_bar.progress((idx + 1) / total_files)
            time.sleep(0.1) # 稍微快一點，因為不用處理圖片

        status_text.text("🎉 全部處理完成！")
        
        # 顯示總結
        st.info(f"📊 結算報告：成功 {success_count} 份 / 失敗 {fail_count} 份")
        
        # 顯示試算表連結 (按鈕形式)
        st.markdown(f"""
            <a href="{SHEET_URL}" target="_blank">
                <button style="
                    background-color: #4CAF50; 
                    border: none;
                    color: white;
                    padding: 15px 32px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                    width: 100%;">
                    📂 點擊這裡查看學生資料 (Google Sheet)
                </button>
            </a>
            """, unsafe_allow_html=True)


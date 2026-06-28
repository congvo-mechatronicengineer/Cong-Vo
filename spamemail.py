import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import requests # Thư viện mới để giao tiếp với Firebase
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- THIẾT LẬP DATABASE ---
# ĐIỀN LINK FIREBASE CỦA BẠN VÀO ĐÂY (Nhớ giữ nguyên chữ /feedback.json ở cuối)
FIREBASE_URL = "https://spamdetectorapp-4ef75-default-rtdb.firebaseio.com/feedback.json"

st.set_page_config(page_title="Universal Spam Detector", page_icon="🤖", layout="wide")
st.title("🤖 Ứng Dụng Lọc Spam AI (Có lưu trữ Đám mây)")

# ==========================================
# 1. TẢI DỮ LIỆU TỪ GITHUB + FIREBASE
# ==========================================
if 'email_db' not in st.session_state:
    all_dfs = []
    
    # A. Đọc dữ liệu gốc từ các file CSV trên GitHub
    dataset_files = glob.glob("*dataset*.csv")
    for file in dataset_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            pass
            
    # B. Tải dữ liệu phản hồi mới nhất từ Firebase Database
    try:
        response = requests.get(FIREBASE_URL)
        if response.status_code == 200 and response.json():
            fb_data = response.json()
            # Biến dữ liệu trên mạng thành bảng
            fb_list = [{"Email": v.get("Email"), "Label": v.get("Label")} for k, v in fb_data.items()]
            all_dfs.append(pd.DataFrame(fb_list))
    except:
        st.sidebar.warning("Đang chạy offline (Không kết nối được Firebase)")

    # C. Trộn chung thành 1 bảng duy nhất
    if all_dfs:
        st.session_state.email_db = pd.concat(all_dfs, ignore_index=True)
    else:
        st.session_state.email_db = pd.DataFrame(columns=["Email", "Label"])

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None

# ==========================================
# 2. GIAO DIỆN (LƯỢC BỎ BỚT CODE HIỂN THỊ ĐỂ BẠN DỄ NHÌN)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗂️ Tổng hợp Dữ liệu", "🧠 Huấn luyện AI", "🔍 Kiểm tra Email"])

with tab1:
    st.header("Kho dữ liệu toàn cầu")
    edited_df = st.data_editor(st.session_state.email_db, num_rows="dynamic", use_container_width=True)
    st.session_state.email_db = edited_df

with tab2:
    if st.button("🚀 Bắt đầu Huấn luyện (Retrain)", type="primary"):
        with st.spinner("AI đang học từ điển tổng hợp..."):
            train_emails = st.session_state.email_db["Email"].astype(str).tolist()
            train_labels = np.array([1 if label == "Spam" else 0 for label in st.session_state.email_db["Label"].tolist()])
            
            vectorizer = CountVectorizer()
            X_train_matrix = vectorizer.fit_transform(train_emails)
            model = MultinomialNB(alpha=1.0)
            model.fit(X_train_matrix, train_labels)
            
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
        st.success("✅ Đã học xong!")

with tab3:
    if st.session_state.model:
        with st.form("predict_form"):
            new_email = st.text_area("Nhập nội dung email cần kiểm tra:")
            submitted = st.form_submit_button("🔍 Quét Email")
            
        if submitted and new_email.strip() != "":
            X_new = st.session_state.vectorizer.transform([new_email])
            prob_spam = st.session_state.model.predict_proba(X_new)[0][1]
            predicted_is_spam = prob_spam > 0.5
            predicted_label = "Spam" if predicted_is_spam else "Ham"
            
            if predicted_is_spam: st.error("🚨 SPAM!")
            else: st.success("✅ HAM")
            
            # --- ĐẨY DỮ LIỆU LÊN ĐÁM MÂY KHI CÓ PHẢN HỒI ---
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Chính xác"):
                    new_data = {"Email": new_email, "Label": predicted_label}
                    # Bắn thẳng dữ liệu lên Google Firebase
                    requests.post(FIREBASE_URL, json=new_data) 
                    # Cập nhật giao diện
                    st.session_state.email_db = pd.concat([st.session_state.email_db, pd.DataFrame([new_data])], ignore_index=True)
                    st.success("Đã lưu vĩnh viễn vào Firebase!")
            with col2:
                if st.button("👎 Sai rồi"):
                    corrected_label = "Ham" if predicted_is_spam else "Spam"
                    new_data = {"Email": new_email, "Label": corrected_label}
                    # Bắn thẳng dữ liệu lên Google Firebase
                    requests.post(FIREBASE_URL, json=new_data)
                    # Cập nhật giao diện
                    st.session_state.email_db = pd.concat([st.session_state.email_db, pd.DataFrame([new_data])], ignore_index=True)
                    st.info("Đã sửa lỗi và lưu vĩnh viễn vào Firebase!")
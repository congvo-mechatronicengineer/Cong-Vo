import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Cấu hình trang web
st.set_page_config(page_title="Spam Detector App", page_icon="📧", layout="wide")
st.title("📧 Ứng Dụng Lọc Spam (Naïve Bayes Classifier)")

# ==========================================
# 1. KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE)
# ==========================================
if 'email_db' not in st.session_state:
    st.session_state.email_db = pd.DataFrame({
        "Email": [
            "Buy cheap medicine now!",
            "Hey, how are you doing today?",
            "Congratulations, you have won a free ticket!",
            "Meeting tomorrow at 10am."
        ],
        "Label": ["Spam", "Ham", "Spam", "Ham"]
    })

if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.vectorizer = None

# ==========================================
# 2. TẠO CÁC TAB GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗂️ Quản lý Dữ liệu", "🧠 Huấn luyện AI", "🔍 Kiểm tra & Phản hồi"])

# --- TAB 1: QUẢN LÝ DỮ LIỆU (CRUD) ---
with tab1:
    st.header("Quản lý Cơ sở dữ liệu Email")
    st.info("💡 Hướng dẫn: Nhấp đúp vào ô để Sửa. Cuộn xuống dòng cuối cùng để Thêm. Chọn hộp thoại bên trái và nhấn phím Delete để Xóa.")
    
    edited_df = st.data_editor(
        st.session_state.email_db,
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Label": st.column_config.SelectboxColumn("Nhãn (Spam/Ham)", options=["Spam", "Ham"], required=True),
            "Email": st.column_config.TextColumn("Nội dung Email", required=True)
        }
    )
    st.session_state.email_db = edited_df

# --- TAB 2: HUẤN LUYỆN MÔ HÌNH NAÏVE BAYES ---
with tab2:
    st.header("Huấn luyện Bộ phân loại Bayes")
    st.write(f"Hiện tại hệ thống đang có **{len(st.session_state.email_db)}** email trong cơ sở dữ liệu.")
    
    if st.button("🚀 Bắt đầu Huấn luyện (Retrain)", type="primary"):
        with st.spinner("Đang huấn luyện AI bằng thuật toán Multinomial Naïve Bayes..."):
            train_emails = st.session_state.email_db["Email"].tolist()
            raw_labels = st.session_state.email_db["Label"].tolist()
            train_labels = np.array([1 if label == "Spam" else 0 for label in raw_labels])
            
            # 1. Tạo "Túi từ vựng" (Bag of words)
            vectorizer = CountVectorizer()
            X_train_matrix = vectorizer.fit_transform(train_emails)
            
            # 2. Áp dụng Multinomial Naive Bayes với Laplace Smoothing (alpha=1.0)
            model = MultinomialNB(alpha=1.0)
            model.fit(X_train_matrix, train_labels)
            
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            
        st.success("✅ Huấn luyện thành công! Thuật toán Bayes đã sẵn sàng.")

# --- TAB 3: DỰ ĐOÁN & THU THẬP PHẢN HỒI ---
with tab3:
    st.header("Kiểm tra Thư Rác (Spam Check)")
    
    if st.session_state.model is None:
        st.warning("⚠️ Mô hình AI chưa được huấn luyện. Vui lòng quay lại Tab 2 để nhấn nút Huấn luyện trước!")
    else:
        with st.form("predict_form", clear_on_submit=False):
            new_email = st.text_area("Nhập nội dung email cần kiểm tra vào đây:", height=150)
            submitted = st.form_submit_button("🔍 Quét Email")
            
        if submitted and new_email.strip() != "":
            # Biến đổi câu thành ma trận đếm từ
            X_new = st.session_state.vectorizer.transform([new_email])
            
            # Dự đoán xác suất với Naive Bayes
            probs = st.session_state.model.predict_proba(X_new)[0]
            prob_spam = probs[1] # Lấy xác suất của nhãn Spam
            
            predicted_is_spam = prob_spam > 0.5
            
            if predicted_is_spam:
                st.error(f"🚨 CẢNH BÁO: Đây là Thư rác (SPAM)! - Xác suất: {prob_spam * 100:.2f}%")
                predicted_label = "Spam"
            else:
                st.success(f"✅ An toàn: Đây là Thư thường (HAM). - Xác suất Spam: {prob_spam * 100:.2f}%")
                predicted_label = "Ham"
            
            # Khu vực thu thập phản hồi
            st.markdown("---")
            st.write("**Kết quả này có chính xác không?**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Chính xác (Giữ nguyên)"):
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [predicted_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.success("Dữ liệu đã được đưa vào CSDL để làm giàu kho từ vựng.")
                    
            with col2:
                if st.button("👎 Sai rồi (Sửa lại nhãn)"):
                    corrected_label = "Ham" if predicted_is_spam else "Spam"
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [corrected_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.info(f"Đã ghi nhận sai sót. Email được lưu lại với nhãn đúng là **{corrected_label}**.")

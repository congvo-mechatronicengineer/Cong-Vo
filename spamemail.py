import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization

# Cấu hình trang web
st.set_page_config(page_title="Spam Detector App", page_icon="📧", layout="wide")
st.title("📧 Ứng Dụng Lọc Spam Email (AI Tự Học)")

# ==========================================
# 1. KHỞI TẠO BỘ NHỚ TẠM (SESSION STATE)
# ==========================================
# Biến session_state giúp web không bị mất dữ liệu khi người dùng bấm nút hay load lại trang
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

if 'tf_model' not in st.session_state:
    st.session_state.tf_model = None
    st.session_state.vectorizer = None

# ==========================================
# 2. TẠO CÁC TAB GIAO DIỆN CHÍNH
# ==========================================
tab1, tab2, tab3 = st.tabs(["🗂️ Quản lý Dữ liệu", "🧠 Huấn luyện AI", "🔍 Kiểm tra & Phản hồi"])

# --- TAB 1: QUẢN LÝ DỮ LIỆU (CRUD) ---
with tab1:
    st.header("Quản lý Cơ sở dữ liệu Email")
    st.info("💡 Hướng dẫn: Bạn có thể nhấp đúp vào ô để **Sửa**. Cuộn xuống dòng cuối cùng để **Thêm** email mới. Chọn hộp thoại bên trái và nhấn phím Delete/Backspace trên bàn phím để **Xóa**.")
    
    # Bảng dữ liệu tương tác (Editable Dataframe)
    edited_df = st.data_editor(
        st.session_state.email_db,
        num_rows="dynamic", # Cho phép người dùng thêm/xóa dòng tự do
        use_container_width=True,
        column_config={
            "Label": st.column_config.SelectboxColumn(
                "Nhãn (Spam/Ham)",
                options=["Spam", "Ham"],
                required=True
            ),
            "Email": st.column_config.TextColumn(
                "Nội dung Email",
                required=True
            )
        }
    )
    # Lưu lại những thay đổi của người dùng vào bộ nhớ
    st.session_state.email_db = edited_df

# --- TAB 2: HUẤN LUYỆN MẠNG NƠ-RON ---
with tab2:
    st.header("Huấn luyện Mô hình TensorFlow")
    st.write(f"Hiện tại hệ thống đang có **{len(st.session_state.email_db)}** email trong cơ sở dữ liệu.")
    
    if st.button("🚀 Bắt đầu Huấn luyện (Retrain)", type="primary"):
        with st.spinner("Đang xây dựng và huấn luyện mạng nơ-ron... Vui lòng đợi!"):
            # Lấy dữ liệu từ bảng
            train_emails = st.session_state.email_db["Email"].tolist()
            raw_labels = st.session_state.email_db["Label"].tolist()
            
            # Chuyển đổi nhãn (Spam = 1, Ham = 0)
            train_labels = np.array([1 if label == "Spam" else 0 for label in raw_labels])
            
            # Khởi tạo Vectorizer
            vectorizer = TextVectorization(max_tokens=2000, output_mode='int', output_sequence_length=15)
            vectorizer.adapt(train_emails)
            
            # Xây dựng mô hình
            model = Sequential([
                vectorizer,
                Embedding(input_dim=2000, output_dim=16),
                GlobalAveragePooling1D(),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(np.array(train_emails), train_labels, epochs=30, verbose=0)
            
            # Lưu mô hình vào session_state để dùng ở Tab 3
            st.session_state.tf_model = model
            st.session_state.vectorizer = vectorizer
            
        st.success("✅ Huấn luyện thành công! Chuyển sang Tab 3 để kiểm tra thử.")

# --- TAB 3: DỰ ĐOÁN & THU THẬP PHẢN HỒI ---
with tab3:
    st.header("Kiểm tra Thư Rác (Spam Check)")
    
    if st.session_state.tf_model is None:
        st.warning("⚠️ Mô hình AI chưa được huấn luyện. Vui lòng quay lại Tab 2 để nhấn nút Huấn luyện trước!")
    else:
        # Form nhập liệu email mới
        with st.form("predict_form", clear_on_submit=False):
            new_email = st.text_area("Nhập nội dung email cần kiểm tra vào đây:", height=150)
            submitted = st.form_submit_button("🔍 Quét Email")
            
        if submitted and new_email.strip() != "":
            # Dự đoán với TensorFlow
            prob = st.session_state.tf_model.predict(np.array([new_email]), verbose=0)[0][0]
            predicted_is_spam = prob > 0.5
            
            # Hiển thị kết quả
            if predicted_is_spam:
                st.error(f"🚨 CẢNH BÁO: Đây là Thư rác (SPAM)! - Xác suất: {prob * 100:.2f}%")
                predicted_label = "Spam"
            else:
                st.success(f"✅ An toàn: Đây là Thư thường (HAM). - Xác suất Spam: {prob * 100:.2f}%")
                predicted_label = "Ham"
            
            # Khu vực thu thập phản hồi
            st.markdown("---")
            st.write("**Kết quả này có chính xác không?**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Chính xác (Giữ nguyên)"):
                    # Thêm vào Database với nhãn máy đoán
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [predicted_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.success("Cảm ơn! Dữ liệu đã được lưu vào CSDL để AI tự học trong tương lai.")
                    
            with col2:
                if st.button("👎 Sai rồi (Sửa lại nhãn)"):
                    # Thêm vào Database với nhãn lật ngược
                    corrected_label = "Ham" if predicted_is_spam else "Spam"
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [corrected_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.info(f"Đã ghi nhận sai sót. Email được lưu lại với nhãn đúng là **{corrected_label}**.")
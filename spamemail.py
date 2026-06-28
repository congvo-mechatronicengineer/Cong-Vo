import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Cấu hình trang web toàn màn hình và đặt tiêu đề
st.set_page_config(page_title="Spam Detector App", page_icon="📧", layout="wide")
st.title("📧 Ứng Dụng Lọc Spam (Naïve Bayes Classifier)")

# ==========================================
# 1. KHỞI TẠO BỘ NHỚ GỐC (SESSION STATE)
# ==========================================
if 'email_db' not in st.session_state:
    try:
        # Tự động đọc hàng ngàn dòng dữ liệu từ file CSV đã tải lên GitHub
        st.session_state.email_db = pd.read_csv("spam_dataset.csv")
    except FileNotFoundError:
        # Phương án cứu cánh: Tạo bảng trống nếu không tìm thấy file để tránh sập ứng dụng web
        st.session_state.email_db = pd.DataFrame(columns=["Email", "Label"])
        st.warning("⚠️ Không tìm thấy file 'spam_dataset.csv'. Hệ thống đang tạm thời dùng bảng dữ liệu trống.")

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
    st.info("💡 Hướng dẫn: Nhấp đúp vào ô để Sửa. Cuộn xuống dòng cuối cùng để Thêm email mới. Chọn hộp thoại ô vuông bên trái và nhấn phím Delete trên bàn phím để Xóa dòng.")
    
    # Bảng dữ liệu Excel tương tác trực quan
    edited_df = st.data_editor(
        st.session_state.email_db,
        num_rows="dynamic", # Cho phép người dùng thêm/xóa dòng tự do trên web
        use_container_width=True,
        column_config={
            "Label": st.column_config.SelectboxColumn("Nhãn (Spam/Ham)", options=["Spam", "Ham"], required=True),
            "Email": st.column_config.TextColumn("Nội dung Email", required=True)
        }
    )
    # Ghi nhận các thao tác Thêm/Sửa/Xóa của người dùng vào bộ nhớ tạm
    st.session_state.email_db = edited_df

# --- TAB 2: HUẤN LUYỆN MÔ HÌNH NAÏVE BAYES ---
with tab2:
    st.header("Huấn luyện Bộ phân loại Bayes")
    st.write(f"Hiện tại hệ thống đang có **{len(st.session_state.email_db)}** email trong cơ sở dữ liệu.")
    
    if st.button("🚀 Bắt đầu Huấn luyện (Retrain)", type="primary"):
        with st.spinner("Đang huấn luyện AI bằng thuật toán Multinomial Naïve Bayes..."):
            # Lấy toàn bộ danh sách dữ liệu mới nhất từ bảng quản lý dữ liệu
            train_emails = st.session_state.email_db["Email"].astype(str).tolist()
            raw_labels = st.session_state.email_db["Label"].tolist()
            train_labels = np.array([1 if label == "Spam" else 0 for label in raw_labels])
            
            # 1. Tạo "Túi từ vựng" (Bag of words) thống kê tần suất xuất hiện
            vectorizer = CountVectorizer()
            X_train_matrix = vectorizer.fit_transform(train_emails)
            
            # 2. Áp dụng thuật toán Multinomial Naive Bayes từ bài giảng Chương 4 (Có kèm Laplace Smoothing alpha=1.0)
            model = MultinomialNB(alpha=1.0)
            model.fit(X_train_matrix, train_labels)
            
            # Lưu trữ "bộ não" đã học xong vào bộ nhớ hệ thống
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            
        st.success("✅ Huấn luyện thành công! Thuật toán Bayes đã sẵn sàng nhận diện.")

# --- TAB 3: DỰ ĐOÁN & THU THẬP PHẢN HỒI ---
with tab3:
    st.header("Kiểm tra Thư Rác (Spam Check)")
    
    if st.session_state.model is None:
        st.warning("⚠️ Mô hình AI chưa được huấn luyện. Vui lòng quay lại Tab 2 để nhấn nút Huấn luyện trước!")
    else:
        # Ô nhập liệu văn bản kiểm tra email rác
        with st.form("predict_form", clear_on_submit=False):
            new_email = st.text_area("Nhập nội dung email cần kiểm tra vào đây:", height=150)
            submitted = st.form_submit_button("🔍 Quét Email")
            
        if submitted and new_email.strip() != "":
            # Biến đổi câu email mới nhập thành ma trận đếm từ theo từ điển đã học
            X_new = st.session_state.vectorizer.transform([new_email])
            
            # Dự đoán và tính toán xác suất phần trăm
            probs = st.session_state.model.predict_proba(X_new)[0]
            prob_spam = probs[1] # Lấy xác suất của nhãn thuộc lớp Spam (Thư rác)
            
            predicted_is_spam = prob_spam > 0.5
            
            # Hiển thị thông báo màu sắc trực quan dựa trên kết quả
            if predicted_is_spam:
                st.error(f"🚨 CẢNH BÁO: Đây là Thư rác (SPAM)! - Xác suất rác: {prob_spam * 100:.2f}%")
                predicted_label = "Spam"
            else:
                st.success(f"✅ An toàn: Đây là Thư thường (HAM). - Xác suất rác: {prob_spam * 100:.2f}%")
                predicted_label = "Ham"
            
            # ==========================================
            # VÒNG LẶP PHẢN HỒI (AI TỰ HỌC)
            # ==========================================
            st.markdown("---")
            st.write("**Kết quả quét này của máy tính có chính xác không?**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Chính xác (Giữ nguyên)"):
                    # Tự động gộp câu vừa quét vào CSDL với nhãn máy đoán để làm giàu kho từ vựng
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [predicted_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.success("Cảm ơn phản hồi! Dữ liệu đã được nạp vào CSDL hệ thống để AI tự học lại trong tương lai.")
                    
            with col2:
                if st.button("👎 Sai rồi (Sửa lại nhãn)"):
                    # Tự động lật ngược nhãn lại cho đúng thực tế và gộp vào CSDL để sửa sai cho thuật toán
                    corrected_label = "Ham" if predicted_is_spam else "Spam"
                    new_row = pd.DataFrame({"Email": [new_email], "Label": [corrected_label]})
                    st.session_state.email_db = pd.concat([st.session_state.email_db, new_row], ignore_index=True)
                    st.info(f"Đã ghi nhận sai sót của máy. Email đã được lưu lại với nhãn chuẩn là **{corrected_label}**.")

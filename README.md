# Hệ Thống Chẩn Đoán Bệnh Tim

Dự án này ứng dụng trí tuệ nhân tạo để phân loại khả năng mắc bệnh tim dựa trên hai nguồn dữ liệu chính: thông tin lâm sàng và hình ảnh siêu âm tim.

## Tính năng chính
- **Chẩn đoán qua chỉ số lâm sàng:** Sử dụng mô hình Ensemble Learning (Stacking giữa XGBoost, LightGBM và Random Forest) dựa trên bộ dữ liệu Cleveland.
- **Phân tích hình ảnh:** Sử dụng mạng nơ-ron tích chập (CNN) với kiến trúc **ResNet-50** để nhận diện dấu hiệu bất thường trên ảnh siêu âm.
- **Giải thích mô hình (XAI):** Tích hợp biểu đồ **SHAP** giúp bác sĩ hiểu rõ tại sao mô hình đưa ra dự đoán (dựa trên các yếu tố như nhịp tim, độ tuổi, loại đau ngực...).
- **Giao diện Web:** Triển khai bằng **Streamlit**, cho phép tương tác trực tiếp dễ dàng.

## Cài đặt
1. Clone repository
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Chạy ứng dụng: `streamlit run app.py`

## Sử dụng

Nhập thông tin lâm sàng hoặc tải lên ảnh siêu âm để nhận kết quả chẩn đoán với độ tin cậy và giải thích chi tiết.
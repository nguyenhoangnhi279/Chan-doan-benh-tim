import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib

st.set_page_config(page_title="Hệ thống Chẩn đoán Bệnh tim", layout="wide")
st.title("Hệ thống Chẩn đoán Bệnh tim")
@st.cache_resource
def load_models():
    tabular_model = joblib.load('heart_disease_stacking_model.pkl')
    scaler = joblib.load('heart_scaler.pkl')
    
    img_model = models.resnet50(pretrained=False)
    num_ftrs = img_model.fc.in_features
    img_model.fc = nn.Linear(num_ftrs, 2)
    img_model.load_state_dict(torch.load('resnet50_heart_disease.pth', map_location=torch.device('cpu')))
    img_model.eval()
    
    return tabular_model, scaler, img_model

try:
    tab_model, scaler, img_model = load_models()
    st.success("Successful!")
except Exception as e:
    st.error(f"Lỗi: {e}")

tab1, tab2 = st.tabs(["Chẩn đoán qua chỉ số lâm sàng", "Chẩn đoán qua hình ảnh siêu âm"])

with tab1:
    st.header("Nhập thông số :")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Tuổi", 1, 120, 50)
        sex = st.selectbox("Giới tính", [1, 0], format_func=lambda x: "Nam" if x==1 else "Nữ")
        cp = st.selectbox("Loại đau ngực (cp)", [0, 1, 2, 3])
    with col2:
        trestbps = st.number_input("Huyết áp", 50, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Đường huyết > 120 mg/dl", [0, 1])
    with col3:
        thalach = st.number_input("Nhịp tim tối đa", 50, 220, 150)
        exang = st.selectbox("Có đau ngực khi vận động không", [0, 1])
        oldpeak = st.number_input("Chỉ số Oldpeak", 0.0, 10.0, 1.0)

    if st.button("Dự đoán qua các chỉ số"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, 0, thalach, exang, oldpeak, 0, 0, 0]])
        input_scaled = scaler.transform(input_data)
        
        prediction = tab_model.predict(input_scaled)
        prob = tab_model.predict_proba(input_scaled)
        
        if prediction[0] == 1:
            st.error(f"CẢNH BÁO: Có khả năng mắc bệnh tim! (Xác suất: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"Thông báo: Chỉ số bình thường. (Xác suất khỏe mạnh: {prob[0][0]*100:.2f}%)")

with tab2:
    st.header("Phân loại ảnh siêu âm tim")
    uploaded_file = st.file_uploader("Xin vui lòng tải ảnh siêu âm lên...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(image).unsqueeze(0)
        
        if st.button("Phân tích hình ảnh"):
            with torch.no_grad():
                output = img_model(img_tensor)
                _, pred = torch.max(output, 1)
                prob = torch.nn.functional.softmax(output, dim=1)
            
            classes = ['Bình thường', 'Có dấu hiệu bệnh']
            result = classes[pred.item()]
            
            if pred.item() == 1:
                st.error(f"Kết quả phân tích ảnh: {result} (Độ tin cậy: {prob[0][1]*100:.2f}%)")
            else:
                st.success(f"Kết quả phân tích ảnh: {result} (Độ tin cậy: {prob[0][0]*100:.2f}%)")
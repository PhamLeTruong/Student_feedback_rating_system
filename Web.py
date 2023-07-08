import streamlit as st
import joblib
import re
from pyvi import ViTokenizer

def Preprocessing(feedback):
    # xóa ký tự kéo dài 
    feedback = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), feedback, flags=re.I)

    # chuẩn hóa chữ thường
    feedback = feedback.lower()

    # xóa ký hiệu teencode
    icons = [':)', ':(', '@@', '<3', ':d', ':3', ':v', ':_', ':p', '>>', ':">', '^^', 'v.v', ':B', ':^', ':v', 'y_y', 'u_u']
    for icon in icons:
        feedback = feedback.replace(icon,' ')
        
    # xóa từ viết tắt mà tác giả đã biến đổi
    acronyms = ['colon', 'smile', 'sad', 'surprise', 'love', 'contemn', 'big', 'smile', 'cc', 
                'small', 'hihi', 'double', 'vdotv','dot', 'fraction', 'csharp']
    for acronym in acronyms:
        feedback = feedback.replace(acronym, ' ')
    
    # xóa ký tự đặc biệt
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for punc in punctuation:
        feedback = feedback.replace(punc,' ') 

    # ViTokenize tách từ
    feedback = ViTokenizer.tokenize(feedback)
    
    return feedback

if 'offensive_words' not in st.session_state:
    with open('vn_offensive_words.txt', encoding='utf8') as f:
        words = f.readlines()
    words = [word[:-1] for word in words]
    st.session_state.offensive_words = words

st.markdown("<h1 style='text-align: center; color: black;'>Sinh Viên Nhập Phản Hồi Đánh Giá</h1>", unsafe_allow_html=True)
feedback = st.text_input('', '', max_chars=250, help='Vui lòng đánh giá bằng tiếng việt và đúng chính tả')
model, tfidf = joblib.load('model.h5')
_, center, _ = st.columns(3)
if feedback:
    st.write('Văn bản phản hồi:', feedback)
    check_offensive = False
    for cmt in feedback.split():
        if cmt in st.session_state.offensive_words:
            check_offensive = True
            break
    if check_offensive:
        center.header('Tiêu Cực')
    else:
        feedback = Preprocessing(feedback)
        feedback = tfidf.transform([feedback])
        pred = model.predict(feedback)
        if pred==0:
            center.header('Tiêu Cực')
        elif pred==1:
            center.header('Trung Tính')
        elif pred==2:
            center.header('Tích cực')

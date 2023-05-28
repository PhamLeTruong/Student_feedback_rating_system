import streamlit as st
import joblib
import re
from pyvi import ViTokenizer
from num2words import num2words
import phunspell
pspell = phunspell.Phunspell('vi_VN')
import pickle

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
    acronyms = ['colonsmile', 'colonsad', 'colonsurprise', 'colonlove', 'colonsmilesmile', 'coloncontemn', 'colonbigsmile', 'coloncc', 
                'colonsmallsmile', 'coloncolon', 'colonlovelove', 'colonhihi', 'doubledot', 'colonsadcolon', 'colonsadcolon', 
                'colondoublesurprise', 'vdotv','dotdotdot', 'fraction', 'csharp', 'dot']
    for acronym in acronyms:
        feedback = feedback.replace(acronym, ' ')
    
    # xóa ký tự đặc biệt
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for punc in punctuation:
        feedback = feedback.replace(punc,' ')

    # ViTokenize tách từ
    feedback = ViTokenizer.tokenize(feedback)
    
    # chuyển đổi số thành chữ
    temp = ''
    for word in feedback.split():
        if word.isnumeric():
            number = num2words(int(word), lang='vi')
            temp += number + ' '   
    # sửa lỗi chính tả
        elif '_' in word or pspell.lookup(word)==True:
            temp += word + ' '
    feedback = temp.lower()

    return feedback

st.header('Mời Sinh Viên Nhập Phản Hồi Đánh Giá')
feedback = st.text_input('', '', max_chars=250, help='Vui lòng đánh giá bằng tiếng việt và đúng chính tả')
model, cv = joblib.load('model.h5')
if feedback:
    st.write('Văn bản phản hồi:', feedback)
    feedback = Preprocessing(feedback)
    feedback = cv.transform([feedback])
    pred = model.predict(feedback)
    _, center, _ = st.columns(3)
    with center:
        if pred==0:
            st.header('Tiêu Cực')
        elif pred==1:
            st.header('Trung Tính')
        elif pred==2:
            st.header('Tích cực')

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle

st.title('Prediksi Serangan Jantung dengan KNN Classification')
st.sidebar.header('Input Data Prediksi Serangan Jantung')

def user_input_features():
    age = st.sidebar.slider('age', 29,77,45)
    sex = st.sidebar.selectbox('sex',(0,1))
    cp = st.sidebar.selectbox('cp',(0,1,2,3))
    trestbps = st.sidebar.slider('trestbps', 94, 200 ,150)
    chol = st.sidebar.slider('chol', 126, 564, 250)
    fbs = st.sidebar.selectbox('fbs',(0,1))
    restecg = st.sidebar.selectbox('restecg',(0, 1, 2))
    thalach = st.sidebar.slider('thalach', 71, 202,150)
    exang = st.sidebar.selectbox('exang',(0,1))
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 3.0)
    slope = st.sidebar.selectbox('slope',(0, 1, 2))
    ca = st.sidebar.selectbox('ca',(0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('thal',(0, 1, 2, 3))
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('Data Input dari User')
st.write(df)

load_predic = pickle.load(open('model.pkl', 'rb'))

prediction = load_predic.predict(df)

st.subheader('Prediksi')
heart_attack_clf = np.array(['Resiko Rendah', 'Resiko Tinggi'])
st.text(heart_attack_clf[prediction])
prediction_proba = load_predic.predict_proba(df)
st.subheader('Prediction Probability')
st.write(prediction_proba)

info = st.button('Informasi Atribut')
if info: 
    st.text(''' 
            Informasi Atribut : 
1. age = umur dalam tahun
2. sex = jenis kelamin (0 = perempuan, 1 = laki - laki)
3. cp = chest pain type / tipe nyeri dada (4 nilai yaitu 0, 1, 2, 3) 
 - Nilai 0: typical angina, 
 - Nilai 1: atypical angina, 
 - Nilai 2: non-anginal pain, 
 - Nilai 3: asymptomatic.
4. trestbps = resting blood pressure / tekanan darah (dalam mm Hg)
5. chol = serum kolesterol dalam mg/dl
6. fbs = fasting blood sugar / gula darah saat puasa : Jika lebih dari 120 mg/dl maka nilainya 1 (true), jika tidak 0 (false)
7. restecg = resting electrocardiographic results / hasil elektrokardiografi saat istirahat (nilai 0,1,2)
 - Nilai 0: normal,
 - Nilai 1: memiliki kelainan gelombang ST-T (inversi gelombang T dan / atau elevasi atau depresi ST> 0,05 mV),
 - Nilai 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes.
8. thalach = maximum heart rate achieved / detak jantung maksimum.
9. exang = exercise induced angina / angina yang diinduksi (1=iya, 0, tidak).
10. oldpeak = ST depression induced by exercise relative to rest / ST Depression disebabkan oleh latihan yang berhubungan dengan istirahat. 
11. slope = the slope of the peak exercise ST segment / kemiringan puncak latihan dari ST Segment.
 - Nilai 0: upsloping / miring ke atas
 - Nilai 1: flat
 - Nilai 2: downsloping / miring ke bawah
12. ca = number of major vessels colored by flourosopy / nomor dari pembuluh utama (0,1,2,3,4)
13. thal = 0 = normal; 1 = fixed defect/cacat tetap; 2 = reversible defect/cacat yang bisa dipulihkan.
14. target: 0= kurang beresiko terkena serangan jantung; 1= lebih beresiko terkena serangan jantung.''')
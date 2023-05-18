import streamlit as st 
import pickle

print('Successfully executed ')

model_lr = pickle.load(open('model_lr.pkl', 'rb'))
model_RMC = pickle.load(open('model_RMC.pkl', 'rb'))
model_xgb = pickle.load(open('model_xgb.pkl', 'rb'))

def predict(model,age,thalach,oldpeak,sex,cp,fbs,exang,slope,ca,thal,trestbps,chol,restecg):
    age = int(age)
    cp = int(cp)
    trestbps = int(trestbps)
    chol = int(chol)
    fbs = int(fbs)
    restecg = int(restecg)
    thalach = int(thalach)
    exang = int(exang)
    oldpeak = int(oldpeak)
    slope = int(slope)
    ca = int(ca)
    thal = int(thal)
    if sex == "female":
        sex = 0
    else:
        sex = 1

    if model=="RandomForestClassifier":
        model = model_RMC
    elif model=="XGBoost":
        model = model_xgb
    else:
        model = model_lr
    prediction=model.predict([[age,thalach,oldpeak,sex,cp,fbs,exang,slope,ca,thal,trestbps,chol,restecg]])
    print(prediction)
    if prediction == 1:
        return 'Heart Disease, Please take treatment'
    else:
        return 'No Heart Disease, Please take Precautions'
    
def main():
    st.title("Heart Prediction")
    html_temp = """
    <div style="background-color:#f8f8f8;padding:20px">
    <h2 style="color:black;text-align:center;">Streamlit Diabetes Predictor </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    model = st.selectbox("Select Model to Run", ["Linear","RandomForestClassifier","XGBoost"])
    age = st.selectbox("Age", list(range(1, 101)), index=0)
    sex = st.selectbox("Sex", ["male", "female"])
    cp = st.selectbox("CP - 3 (Lowest risk) - 0 (Highest Risk)", ["0","1","2","3"])
    trestbps = st.text_input("trestbps - 3 (Lowest risk) - 0 (Highest Risk)", "Type Here")
    chol = st.text_input("chol - 3 (Lowest risk) - 0 (Highest Risk)", "Type Here")
    fbs=st.selectbox("FBS - 0 - 1", ["0","1"])
    restecg = st.selectbox("restecg - 3 (Lowest risk) - 0 (Highest Risk)", ["0","1","2","3"])
    thalach=st.text_input("Thalach - Give input between 20-500","Type Here")
    exang=st.selectbox("Exang - 0 - 1", ["0","1"])
    oldpeak=st.selectbox("Old Peak", [round(x * 0.1, 1) for x in range(0, 101)])
    slope =st.selectbox("slope - 0 (Lowest risk) - 2 (Highest Risk)",["0","1","2"])
    ca=st.selectbox("CA - 0 (Lowest risk) - 3 (Highest Risk)", ["0","1","2","3"])
    thal=st.selectbox("Thal - 0 (Lowest risk) - 2 (Highest Risk)", ["0","1","2"])
    

    result=""
    if st.button("Predict"):
        result=predict(model,age,thalach,oldpeak,sex,cp,fbs,exang,slope,ca,thal,trestbps,chol,restecg)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
main()

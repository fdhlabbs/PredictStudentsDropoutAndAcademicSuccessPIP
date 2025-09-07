# app.py


#import
#--------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import preprocess as pp

from sklearn.model_selection import (
    GridSearchCV,
    train_test_split)

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler)

from sklearn.neural_network import MLPClassifier

from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
#--------------------------------------------------

# ------------------------- Page Setup -------------------------
st.set_page_config(page_title="Student Outcome Predictor v3", layout="wide")

# Add banner
st.image("img/banner.png", use_container_width=True)

st.title("🎓 Predict Student Dropout / Success")

# ------------------------- Model Selection -------------------------
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["RandomForest", "ANN", "HistGradientBoosting"])

# ------------------------- Helpers -------------------------
def get_key(mapping, value):
    for k, v in mapping.items():
        if v == value:
            return k
    return None

inv_conv = {v: k for k, v in pp.conversion_dict.items()}
# ------------------------- User Input Form -------------------------

with st.form("input_form"):
    st.subheader("🧑 Personal Info")
    maritalStatus = st.selectbox(inv_conv["maritalStatus"], list(pp.maritalStatus.values()))
    nationality = st.selectbox(inv_conv["nationality"], list(pp.nationality.values()))
    displaced = st.selectbox(inv_conv["displaced"], list(pp.displaced.values()))
    educationalSpecialNeeds = st.selectbox(inv_conv["educationalSpecialNeeds"], list(pp.educationalSpecialNeeds.values()))
    gender = st.selectbox(inv_conv["gender"], list(pp.gender.values()))
    international = st.selectbox(inv_conv["international"], list(pp.international.values()))

    st.divider()
    st.subheader("👪 Family Info")
    motherQualification = st.selectbox(inv_conv["motherQualification"], list(pp.motherQualification.values()))
    fatherQualification = st.selectbox(inv_conv["fatherQualification"], list(pp.fatherQualification.values()))
    motherOccupation = st.selectbox(inv_conv["motherOccupation"], list(pp.motherOccupation.values()))
    fatherOccupation = st.selectbox(inv_conv["fatherOccupation"], list(pp.fatherOccupation.values()))

    st.divider()
    st.subheader("📚 Education Info")
    applicationMode = st.selectbox(inv_conv["applicationMode"], list(pp.applicationMode.values()))
    applicationOrder = st.number_input(inv_conv["applicationOrder"], min_value=1)
    course = st.selectbox(inv_conv["course"], list(pp.course.values()))
    daytimeEveningAttendance = st.selectbox(inv_conv["daytimeEveningAttendance"], list(pp.daytimeEveningAttendance.values()))
    previousQualification = st.selectbox(inv_conv["previousQualification"], list(pp.previousQualification.values()))
    previousQualificationGrade = st.number_input(inv_conv["previousQualificationGrade"], min_value=0.0, max_value=200.0)
    admissionGrade = st.number_input(inv_conv["admissionGrade"], min_value=0.0, max_value=200.0)
    debtor = st.selectbox(inv_conv["debtor"], list(pp.debtor.values()))
    tuitionFeesUpToDate = st.selectbox(inv_conv["tuitionFeesUpToDate"], list(pp.tuitionFeesUpToDate.values()))
    scholarshipHolder = st.selectbox(inv_conv["scholarshipHolder"], list(pp.scholarshipHolder.values()))
    ageAtEnrollment = st.number_input(inv_conv["ageAtEnrollment"], min_value=15, value =18, max_value=60)

    st.divider()
    st.subheader("🏫 1st Sem Info")
    curricularUnits1stSemCredited = st.number_input(inv_conv["curricularUnits1stSemCredited"], min_value=0)
    curricularUnits1stSemEnrolled = st.number_input(inv_conv["curricularUnits1stSemEnrolled"], min_value=0)
    curricularUnits1stSemEvaluations = st.number_input(inv_conv["curricularUnits1stSemEvaluations"], min_value=0)
    curricularUnits1stSemApproved = st.number_input(inv_conv["curricularUnits1stSemApproved"], min_value=0)
    curricularUnits1stSemGrade = st.number_input(inv_conv["curricularUnits1stSemGrade"], min_value=0.0)
    curricularUnits1stSemWithoutEvaluations = st.number_input(inv_conv["curricularUnits1stSemWithoutEvaluations"], min_value=0)

    st.divider()
    # ------------------------- 2nd Sem Info -------------------------
    st.subheader("🏫 2nd Sem Info")
    curricularUnits2ndSemCredited = st.number_input(inv_conv["curricularUnits2ndSemCredited"], min_value=0)
    curricularUnits2ndSemEnrolled = st.number_input(inv_conv["curricularUnits2ndSemEnrolled"], min_value=0)
    curricularUnits2ndSemEvaluations = st.number_input(inv_conv["curricularUnits2ndSemEvaluations"], min_value=0)
    curricularUnits2ndSemApproved = st.number_input(inv_conv["curricularUnits2ndSemApproved"], min_value=0)
    curricularUnits2ndSemGrade = st.number_input(inv_conv["curricularUnits2ndSemGrade"], min_value=0.0)
    curricularUnits2ndSemWithoutEvaluations = st.number_input(inv_conv["curricularUnits2ndSemWithoutEvaluations"], min_value=0)

    st.divider()
    st.subheader("📈 Economy Situation at Enrollment")
    unemploymentRate = st.number_input(inv_conv["unemploymentRate"], min_value=0.0, max_value=50.0)
    inflationRate = st.number_input(inv_conv["inflationRate"], min_value=-10.0, value=0.0, max_value=10.0)
    gdp = st.number_input(inv_conv["gdp"], min_value=-10.0, value=0.0, max_value=10.0)

    st.divider()
    submitted = st.form_submit_button("Predict")

    #age_input = st.number_input("Age at Enrollment", min_value=15, max_value=60, value=18, step=1)
    
# ------------------------- Predict -------------------------
if submitted:
    input_dict = {
        # Personal Info
        "maritalStatus": get_key(pp.maritalStatus, maritalStatus),
        "nationality": get_key(pp.nationality, nationality),
        "displaced": get_key(pp.displaced, displaced),
        "educationalSpecialNeeds": get_key(pp.educationalSpecialNeeds, educationalSpecialNeeds),
        "gender": get_key(pp.gender, gender),
        "international": get_key(pp.international, international),

        # Family Info
        "motherQualification": get_key(pp.motherQualification, motherQualification),
        "fatherQualification": get_key(pp.fatherQualification, fatherQualification),
        "motherOccupation": get_key(pp.motherOccupation, motherOccupation),
        "fatherOccupation": get_key(pp.fatherOccupation, fatherOccupation),

        # Education Info
        "applicationMode": get_key(pp.applicationMode, applicationMode),
        "applicationOrder": applicationOrder,
        "course": get_key(pp.course, course),
        "daytimeEveningAttendance": get_key(pp.daytimeEveningAttendance, daytimeEveningAttendance),
        "previousQualification": get_key(pp.previousQualification, previousQualification),
        "previousQualificationGrade": previousQualificationGrade,
        "admissionGrade": admissionGrade,
        "debtor": get_key(pp.debtor, debtor),
        "tuitionFeesUpToDate": get_key(pp.tuitionFeesUpToDate, tuitionFeesUpToDate),
        "scholarshipHolder": get_key(pp.scholarshipHolder, scholarshipHolder),
        "ageAtEnrollment": ageAtEnrollment,

        # 1st Sem
        "curricularUnits1stSemCredited": curricularUnits1stSemCredited,
        "curricularUnits1stSemEnrolled": curricularUnits1stSemEnrolled,
        "curricularUnits1stSemEvaluations": curricularUnits1stSemEvaluations,
        "curricularUnits1stSemApproved": curricularUnits1stSemApproved,
        "curricularUnits1stSemGrade": curricularUnits1stSemGrade,
        "curricularUnits1stSemWithoutEvaluations": curricularUnits1stSemWithoutEvaluations,

        # 2nd Sem
        "curricularUnits2ndSemCredited": curricularUnits2ndSemCredited,
        "curricularUnits2ndSemEnrolled": curricularUnits2ndSemEnrolled,
        "curricularUnits2ndSemEvaluations": curricularUnits2ndSemEvaluations,
        "curricularUnits2ndSemApproved": curricularUnits2ndSemApproved,
        "curricularUnits2ndSemGrade": curricularUnits2ndSemGrade,
        "curricularUnits2ndSemWithoutEvaluations": curricularUnits2ndSemWithoutEvaluations,

        # Economy
        "unemploymentRate": unemploymentRate,
        "inflationRate": inflationRate,
        "gdp": gdp
    }

    stud = pd.DataFrame([input_dict])


    st.title("🧠 Prediction Result")

# ------------------------- RandomForest -------------------------
    pred = None
    proba = None
    
    if model_choice == "RandomForest":
        #process stud
        stud = pp.rfPreProc(stud)
    
        #reindex
        rf_columns = joblib.load("models/model_rf_columns.pkl")
        stud = stud.reindex(columns=rf_columns, fill_value=0)
    
        #reimpute
        rf_imputer = joblib.load("models/model_rf_imputer.pkl")
        stud_imp = pd.DataFrame(rf_imputer.transform(stud), columns=rf_columns)
    
        #predict
        model_rf = joblib.load("models/model_rf.pkl")
        pred = model_rf.predict(stud_imp)[0]
        proba = model_rf.predict_proba(stud_imp)[0]

# ------------------------- ANN -------------------------
    elif model_choice == "ANN":
        #get scale
        pp.scale_ann = joblib.load("models/model_ann_scalers.pkl")
    
        #process stud
        stud = pp.annPreProc(stud)
    
        #reindex
        preproc_columns = joblib.load("models/model_ann_columns.pkl")
        stud = stud.reindex(columns=preproc_columns, fill_value=0)

        stud = stud.fillna(0)
        #predict
        model_ann = joblib.load("models/model_ann.pkl")
        pred = model_ann.predict(stud)[0]
        proba = model_ann.predict_proba(stud)[0]

# ------------------------- HistGradientBoosting -------------------------
    elif model_choice == "HistGradientBoosting":
        #process stud
        stud = pp.hgbPreProc(stud)
    
        #reindex
        hgb_columns = joblib.load("models/model_hgb_columns.pkl")
        stud = stud.reindex(columns=hgb_columns, fill_value=0)
    
        #reimpute
        hgb_imputer = joblib.load("models/model_hgb_imputer.pkl")
        stud = pd.DataFrame(hgb_imputer.transform(stud), columns=hgb_columns)
    
        #predict
        model_hgb = joblib.load("models/model_hgb.pkl")
        pred = model_hgb.predict(stud)[0]
        proba = model_hgb.predict_proba(stud)[0]
# ------------------------- else -------------------------
    else:
        st.error("❌ Invalid model selection.")
# ------------------------- success -------------------------

    st.success(f"🎯 Prediction: `{pp.targetMapReverse[pred]}`")
    st.write("📊 Probability:", {pp.targetMapReverse[i]: round(p * 100, 2) for i, p in enumerate(proba)})
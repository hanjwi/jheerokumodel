from flask import Flask, request, Response, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

master = pd.read_csv('../Data/healthcare-dataset-stroke-data.csv')
master['bmi'] = master['bmi'].fillna(master.bmi.mean())
category_data = master[['work_type','Residence_type','smoking_status']].values
gender = master[['gender']].values
married = master[['ever_married']].values
ohe = OneHotEncoder()
le_gender = LabelEncoder()
le_married = LabelEncoder()
category = ohe.fit_transform(category_data).toarray()
category_df = pd.DataFrame(category)
category_df.columns = ohe.get_feature_names()
gender[:,0] = le_gender.fit_transform(gender[:,0])
married[:,0] = le_married.fit_transform(married[:,0])
gender_df = pd.DataFrame(gender, columns = ['gender'])
married_df = pd.DataFrame(married, columns = ['married'])
numeric = master.iloc[:,[2,8,9]]
age_sc = StandardScaler()
glucose_sc = StandardScaler()
bmi_sc = StandardScaler()
numeric['age'] = age_sc.fit_transform(numeric[['age']])
numeric['avg_glucose_level'] = glucose_sc.fit_transform(numeric[['avg_glucose_level']])
numeric['bmi'] = bmi_sc.fit_transform(numeric[['bmi']])
already_labeled = master.iloc[:,[3,4]]
x_final = pd.concat([category_df,gender_df, married_df, already_labeled, numeric], axis = 1)
y_final = master['stroke']
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(x_final, y_final)

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    data_age = np.array([data['age']]).reshape(1,-1)
    data_age = np.array(age_sc.transform(data_age))
    data_glucose = np.array([data['avg_glucose_level']]).reshape(1,-1)
    data_glucose = np.array(glucose_sc.transform(data_glucose))
    data_bmi = np.array([data['bmi']]).reshape(1,-1)
    data_bmi = np.array(bmi_sc.transform(data_bmi))
    data_gender = np.array([data['gender']]).astype('object')
    data_gender = le_gender.transform(data_gender)
    data_married = np.array([data['married']])
    data_married = le_married.transform(data_married)
    data_cat = np.array([data['work_type'], data['Residence_type'], data['smoking_status']]).reshape(-1,3)
    data_cat = ohe.transform(data_cat).toarray()
    hypertension_data = data['hypertension']
    heartdesease_data = data['heart_desease']
    data_ot = np.array([[hypertension_data, heartdesease_data]])
    data_f = np.column_stack((data_cat, data_gender, data_married, data_ot, data_age, data_glucose, data_bmi))
    data_f = pd.DataFrame(data_f, dtype=object)
    prediction = rfc.predict(data_f)
    prediction = prediction.astype('object')
    if prediction ==0:
        return Response(json.dumps("No"))
    else: return Response(json.dumps("Yes"))

import pandas as pd

def preprocess_input(form):
    # Chuyển form từ HTML thành vector input cho model
    features = [
        int(form['gender']),
        int(form['age']),
        int(form['hypertension']),
        int(form['heart_disease']),
        int(form['ever_married']),
        int(form['work_type']),
        int(form['Residence_type']),
        float(form['avg_glucose_level']),
        float(form['bmi']),
        int(form['smoking_status'])
    ]
    return features

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def parse_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['id'], errors='ignore')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df['smoking_status'] = df['smoking_status'].replace('Unknown', 'unknown')

    cat_map = {
        'gender': {'Male': 1, 'Female': 0, 'Other': 2},
        'ever_married': {'Yes': 1, 'No': 0},
        'work_type': {'Private': 2, 'Self-employed': 3, 'Govt_job': 1, 'children': 0, 'Never_worked': 4},
        'Residence_type': {'Urban': 1, 'Rural': 0},
        'smoking_status': {'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'unknown': 0}
    }

    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df[[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]]

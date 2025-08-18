import pandas as pd
import pickle
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load dữ liệu
df = pd.read_csv('train/healthcare-dataset-stroke-data.csv')  
print("✅ Dữ liệu đã load thành công!")

# 2. Tiền xử lý dữ liệu
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 2.2 Xử lý missing value cột bmi bằng XGBoost Regressor
df_bmi_train = df[df['bmi'].notnull()]
df_bmi_missing = df[df['bmi'].isnull()]
bmi_features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                'work_type', 'Residence_type', 'avg_glucose_level', 'smoking_status']

xgb_reg = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_reg.fit(df_bmi_train[bmi_features], df_bmi_train['bmi'])
predicted_bmi = xgb_reg.predict(df_bmi_missing[bmi_features])
df.loc[df['bmi'].isnull(), 'bmi'] = predicted_bmi

print("✅ Đã dự đoán và điền giá trị thiếu của cột BMI bằng XGBoost!")

# 3. Chuẩn bị feature và label
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# 4. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Đã chia tập train/test!")

# 5. Xử lý mất cân bằng bằng SMOTE trên tập train
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"✅ Sau SMOTE, số lượng lớp 0: {(y_train_res==0).sum()}, lớp 1: {(y_train_res==1).sum()}")

# 6. Train Random Forest trên dữ liệu đã xử lý mất cân bằng
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_res, y_train_res)
print("✅ Mô hình Random Forest phân loại đã được train!")

# 7. Đánh giá mô hình trên test (chưa qua SMOTE)
y_pred = rf_clf.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Classification report:\n", classification_report(y_test, y_pred))

# 8. Lưu mô hình và encoders
os.makedirs("model", exist_ok=True)
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

with open('model/stroke_rf.pkl', 'wb') as f:
    pickle.dump((rf_clf, encoders), f)

print("✅ Mô hình Random Forest và encoders đã được lưu tại model/stroke_rf.pkl")

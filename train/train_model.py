import pandas as pd
import pickle
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dữ liệu
df = pd.read_csv('train/healthcare-dataset-stroke-data.csv')  
print("✅ Dữ liệu đã load thành công!")

# 2. Tiền xử lý dữ liệu

# 2.1 Encode các cột dạng categorical
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 2.2 Xử lý missing value cho cột 'bmi' bằng XGBoost Regressor
# Chia dữ liệu thành 2 phần: có bmi và thiếu bmi
df_bmi_train = df[df['bmi'].notnull()]
df_bmi_missing = df[df['bmi'].isnull()]

# Các đặc trưng dùng để dự đoán bmi
bmi_features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                'work_type', 'Residence_type', 'avg_glucose_level', 'smoking_status']

# Train XGBoost Regressor
xgb_reg = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_reg.fit(df_bmi_train[bmi_features], df_bmi_train['bmi'])

# Dự đoán giá trị thiếu
predicted_bmi = xgb_reg.predict(df_bmi_missing[bmi_features])

# Gán lại vào DataFrame
df.loc[df['bmi'].isnull(), 'bmi'] = predicted_bmi

print("✅ Đã dự đoán và điền giá trị thiếu của cột BMI bằng XGBoost!")

# 3. Tách feature và label cho bài toán dự đoán đột quỵ
X = df.drop(columns=['id', 'stroke'])  # Bỏ cột 'id'
y = df['stroke']

# 4. Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Đã chia tập train/test!")

# 5. Train mô hình Random Forest phân loại đột quỵ
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
print("✅ Mô hình Random Forest phân loại đã được train!")

# 6. Đánh giá mô hình
y_pred = rf_clf.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Classification report:\n", classification_report(y_test, y_pred))

# 7. Tạo thư mục lưu mô hình nếu chưa có
os.makedirs("model", exist_ok=True)

# 8. Lưu mô hình Random Forest
with open('model/stroke_rf.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

print("✅ Mô hình Random Forest đã được lưu tại model/stroke_rf.pkl")

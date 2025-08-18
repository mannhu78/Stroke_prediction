from flask import Flask, render_template, request, redirect, url_for, session, flash,send_file
from werkzeug.security import generate_password_hash, check_password_hash
import pymysql
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from db_config import get_db_connection
from sklearn.preprocessing import LabelEncoder
from flask_login import login_required
import os
import uuid
import base64
from flask import request
from PIL import Image
import io
import joblib
import cv2
import mediapipe as mp
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Giới hạn 10MB
app.secret_key = 'stroke_secret_key'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
# Load model
model,encoders = pickle.load(open('model/stroke_rf.pkl', 'rb'))
face_model = joblib.load('model/face_model.pkl')
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
# In ra các class để kiểm tra
for col in categorical_cols:
    print(f"{col}: {encoders[col].classes_}")
# ---------------------
# Helper
# ---------------------
def save_prediction_to_db(user_id, input_data, result):
    connection = get_db_connection()
    with connection.cursor() as cursor:
        columns = ', '.join(input_data.keys())
        values = tuple(input_data.values())
        sql = f"""
        INSERT INTO prediction (user_id, {columns}, result, timestamp)
        VALUES (%s, {', '.join(['%s']*len(values))}, %s, %s)
        """
        cursor.execute(sql, (user_id, *values, result, datetime.now()))
        connection.commit()
    connection.close()

# ---------------------
# Routes
# ---------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            if cursor.fetchone():
                flash('Email đã tồn tại.')
                return redirect(url_for('register'))
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, password))
            connection.commit()
        connection.close()
        flash('Đăng ký thành công. Vui lòng đăng nhập.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['email'] = user[1]
                return redirect(url_for('about'))
            else:
                flash('Email hoặc mật khẩu không đúng.')
        connection.close()
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    result = None
    advice_title = ""
    advice_items = []
    
    if request.method == 'POST':
        try:
            # Lấy input dạng string hoặc số phù hợp
            input_data = {
                'gender': request.form['gender'],
                'age': float(request.form['age']),
                'hypertension': int(request.form['hypertension']),
                'heart_disease': int(request.form['heart_disease']),
                'ever_married': request.form['ever_married'],
                'work_type': request.form['work_type'],
                'Residence_type': request.form['Residence_type'],
                'avg_glucose_level': float(request.form['avg_glucose_level']),
                'smoking_status': request.form['smoking_status'],
            }
            
            # Tính BMI từ height(cm), weight(kg)
            height_cm = float(request.form['height'])
            weight_kg = float(request.form['weight'])
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            input_data['bmi'] = round(bmi, 1)

            

            # Chuyển dữ liệu thành DataFrame theo đúng thứ tự cột khi train
            feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
            df_input = pd.DataFrame([input_data], columns=feature_cols)

            # Dự đoán xác suất đột quỵ (class 1)
            proba = model.predict_proba(df_input)[0][1]
            result = round(proba * 100, 2)

            # Ghi kết quả vào DB hoặc file nếu cần
            save_prediction_to_db(session['user_id'], input_data, result)

            # Gợi ý phòng ngừa
            if result >= 70:
                advice_title = "⚠️ Nguy cơ cao!"
                advice_items = [
                    "Khám sức khỏe chuyên sâu càng sớm càng tốt.",
                    "Kiểm soát huyết áp, đường huyết, mỡ máu dưới sự theo dõi của bác sĩ.",
                    "Tuyệt đối tránh hút thuốc, rượu bia.",
                    "Thay đổi lối sống khẩn cấp: tập thể dục, ăn uống lành mạnh, giảm stress."
                ]
            elif result >= 30:
                advice_title = "🔶 Nguy cơ trung bình"
                advice_items = [
                    "Theo dõi sức khỏe định kỳ 3-6 tháng/lần.",
                    "Giảm thiểu căng thẳng và tăng cường vận động thể chất.",
                    "Điều chỉnh chế độ ăn: ít muối, ít dầu mỡ, tăng rau xanh.",
                    "Hạn chế hút thuốc, rượu bia nếu có."
                ]
            else:
                advice_title = "✅ Nguy cơ thấp"
                advice_items = [
                    "Tiếp tục duy trì chế độ sinh hoạt lành mạnh.",
                    "Tập thể dục đều đặn (30 phút/ngày, 5 ngày/tuần).",
                    "Khám sức khỏe định kỳ để tầm soát sớm các yếu tố nguy cơ.",
                    "Tránh hút thuốc lá và kiểm soát cân nặng."
                ]

        except Exception as e:
            print("Error during prediction:", e)

    return render_template('predict.html', result=result, advice_title=advice_title, advice_items=advice_items)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print("📥 Vào route /upload")
    
    if request.method == 'POST':
        print("📨 Phương thức POST")
        if 'csv_file' not in request.files:
            print("❌ Không có csv_file trong request")
            flash('Không có file được chọn!', 'danger')
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            print("❌ Tên file trống")
            flash('Chưa chọn file!', 'danger')
            return redirect(request.url)

        try:
            print("📄 Đang đọc file CSV")
            df = pd.read_csv(file)
            print("📊 Dữ liệu đọc được:")
            print(df.head())

            # Encode các cột giống như khi train
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            for col in categorical_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

            df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

            # Dự đoán
            X_input = df.drop(columns=['id', 'stroke'], errors='ignore')
            print("✅ Dữ liệu đầu vào sau xử lý:")
            print(X_input.head())

            preds = model.predict_proba(X_input)[:, 1] * 100  # tỉ lệ %

            # In kết quả
            print("🧠 Dự đoán tỉ lệ đột quỵ:")
            print(preds)

            # Chuẩn bị kết quả hiển thị
            results = []
            for i in range(len(preds)):
                row_data = {}

                # Đặt ID đầu tiên
                if 'id' in df.columns:
                    row_data['ma_benh_nhan'] = df.iloc[i]['id']
                else:
                    row_data['id'] = 'Không rõ'

                # Thêm các thông tin cần hiển thị
                row_data['gioi_tinh'] = X_input.iloc[i]['gender']
                row_data['tuoi'] = X_input.iloc[i]['age']
                row_data['huyet_ap'] = X_input.iloc[i]['hypertension']
                row_data['benh_tim'] = X_input.iloc[i]['heart_disease']
                row_data['hon_nhan'] = X_input.iloc[i]['ever_married']
                row_data['cong_viec'] = X_input.iloc[i]['work_type']
                row_data['noi_o'] = X_input.iloc[i]['Residence_type']
                row_data['muc_duong_huyet'] = X_input.iloc[i]['avg_glucose_level']
                row_data['bmi'] = X_input.iloc[i]['bmi']
                row_data['hut_thuoc'] = X_input.iloc[i]['smoking_status']

                # Nhãn thực tế nếu có
                row_data['nguy_co_dot_quy'] = df.iloc[i]['stroke'] if 'stroke' in df.columns else 'Không rõ'

                # Kết quả dự đoán
                row_data['phan_tram_du_doan'] = round(preds[i], 2)

                results.append(row_data)

            print("✅ RESULTS to render:")
            print(results)

            # Chuyển kết quả sang DataFrame
            df_result = pd.DataFrame(results)

            # Tạo tên file duy nhất
            filename = f"result_{uuid.uuid4().hex}.csv"
            output_path = os.path.join('static/results', filename)
            df_result.to_csv(output_path, index=False)

            # Gửi file name vào template
            return render_template('upload_result.html', 
                                   results=results, 
                                   download_link=url_for('download_result', filename=filename),
                                   stroke_count=sum(1 for row in results if row['nguy_co_dot_quy'] == 1),
                                   no_stroke_count=sum(1 for row in results if row['nguy_co_dot_quy'] == 0))

        except Exception as e:
            print("❗Lỗi xảy ra:", e)
            flash(f'Đã xảy ra lỗi: {str(e)}', 'danger')
            return redirect(request.url)

    print("➡️ Truy cập upload bằng GET")
    return render_template('upload.html')


@app.route('/download_result/<filename>')
def download_result(filename):
    file_path = os.path.join('static/results', filename)
    return send_file(file_path, as_attachment=True)


@app.route('/history', methods=['GET', 'POST'])
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_date = None
    if request.method == 'POST':
        selected_date = request.form.get('selected_date')

    connection = get_db_connection()
    with connection.cursor(pymysql.cursors.DictCursor) as cursor:
        if selected_date:
            query = "SELECT * FROM prediction WHERE user_id=%s AND DATE(timestamp)=%s ORDER BY timestamp DESC"
            cursor.execute(query, (session['user_id'], selected_date))
        else:
            query = "SELECT * FROM prediction WHERE user_id=%s ORDER BY timestamp DESC"
            cursor.execute(query, (session['user_id'],))
        records = cursor.fetchall()
    connection.close()
    return render_template('history.html', history=records, selected_date=selected_date)





# Folder lưu ảnh
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/face_analysis')
def face_analysis_page():
    return render_template('face_analysis.html')

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    file = request.files.get('image')
    if not file:
        return render_template('face_analysis.html', error="Vui lòng chọn ảnh.")

    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return render_template('face_analysis.html', error="Không phát hiện khuôn mặt.")

        annotated = image.copy()
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )

        # Lấy điểm mép miệng
        lm = landmarks.landmark
        left_mouth = lm[61]
        right_mouth = lm[291]
        top_lip = lm[13]
        bottom_lip = lm[14]

        mouth_diff = abs(left_mouth.y - right_mouth.y)
        lip_height = abs(top_lip.y - bottom_lip.y)

        diagnosis = "✅ Miệng đối xứng bình thường."
        if mouth_diff > 0.02:
            diagnosis = "⚠️ Có thể có dấu hiệu méo miệng - cần kiểm tra thêm."

        result_img = 'uploads/result_' + filename
        out_path = os.path.join('static', result_img)
        cv2.imwrite(out_path, annotated)

        return render_template('face_analysis.html',
                               result_img=result_img,
                               diagnosis=diagnosis,
                               mouth_diff=round(mouth_diff, 4),
                               lip_height=round(lip_height, 4))
    
@app.route('/analyze-face-webcam', methods=['POST'])
def analyze_face_webcam():
    data_url = request.form.get('image_data')
    if not data_url:
        return render_template('face_analysis.html', error="Không nhận được ảnh từ webcam.")

    # Tách base64
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(img_data)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Tiếp tục giống như route analyze_face:
    # Xử lý bằng Mediapipe, chẩn đoán, vẽ lại ảnh, lưu ảnh

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return render_template('face_analysis.html', error="Không phát hiện khuôn mặt.")

        annotated = image_bgr.copy()
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )

        # Lấy thông số phân tích như trước
        lm = landmarks.landmark
        left_mouth = lm[61]
        right_mouth = lm[291]
        top_lip = lm[13]
        bottom_lip = lm[14]

        mouth_diff = abs(left_mouth.y - right_mouth.y)
        lip_height = abs(top_lip.y - bottom_lip.y)

        diagnosis = "✅ Miệng đối xứng bình thường."
        if mouth_diff > 0.02:
            diagnosis = "⚠️ Có thể có dấu hiệu méo miệng - cần kiểm tra thêm."

        # Lưu ảnh đã xử lý
        filename = 'webcam_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.png'
        out_path = os.path.join('static/uploads', filename)
        cv2.imwrite(out_path, annotated)

        # Ghi vào DB nếu cần (giống phần trước)
        if 'user_id' in session:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                input_data = f"Mouth diff: {round(mouth_diff,4)}, Lip height: {round(lip_height,4)}"
                cursor.execute("INSERT INTO prediction (user_id, input_data, result) VALUES (%s, %s, %s)",
                               (session['user_id'], input_data, diagnosis))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                print("DB Error:", e)

        return render_template('face_analysis.html',
                               result_img='uploads/' + filename,
                               diagnosis=diagnosis,
                               mouth_diff=round(mouth_diff, 4),
                               lip_height=round(lip_height, 4))


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "Không có file!"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Đọc dữ liệu
        df = pd.read_csv(filepath)

        # Trực quan hóa
        sns.set(style="whitegrid")
        img_paths = []

        # 1. Tuổi theo giới tính
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x='age', hue='gender', bins=30, kde=True)
        plt.title('Phân bố tuổi theo giới tính')
        img1_path = os.path.join(STATIC_FOLDER, 'plot_age_gender.png')
        plt.savefig(img1_path)
        img_paths.append('plot_age_gender.png')
        plt.close()

        # 4. Tỉ lệ đột quỵ
        plt.figure(figsize=(6, 6))
        stroke_counts = df['stroke'].value_counts()
        labels = ['Không đột quỵ', 'Đột quỵ']
        plt.pie(stroke_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
        plt.title('Tỉ lệ đột quỵ trong tập dữ liệu')
        img4_path = os.path.join(STATIC_FOLDER, 'plot_stroke_rate.png')
        plt.savefig(img4_path)
        img_paths.append('plot_stroke_rate.png')
        plt.close()

        # 2. Nghề nghiệp & đột quỵ
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='work_type', hue='stroke')
        plt.title('Tỉ lệ đột quỵ theo công việc')
        plt.xticks(rotation=45)
        img2_path = os.path.join(STATIC_FOLDER, 'plot_work_stroke.png')
        plt.savefig(img2_path)
        img_paths.append('plot_work_stroke.png')
        plt.close()

        # 3. Ma trận tương quan
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=['float64', 'int64']).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Ma trận tương quan các đặc trưng số')
        img3_path = os.path.join(STATIC_FOLDER, 'plot_corr.png')
        plt.savefig(img3_path)
        img_paths.append('plot_corr.png')
        plt.close()
         # 5. Tỉ lệ đột quỵ theo bệnh tim
        plt.figure(figsize=(6, 4))
        sns.barplot(data=df, x='heart_disease', y='stroke', estimator=lambda x: sum(x)/len(x))
        plt.xticks([0, 1], ['Không bệnh tim', 'Có bệnh tim'])
        plt.title('Tỉ lệ đột quỵ theo tình trạng bệnh tim')
        plt.ylabel('Tỉ lệ đột quỵ')
        plt.ylim(0, 1)
        img5_path = os.path.join(STATIC_FOLDER, 'plot_stroke_heart.png')
        plt.savefig(img5_path)
        img_paths.append('plot_stroke_heart.png')
        plt.close()

        # 6. Tỉ lệ đột quỵ theo tình trạng hút thuốc
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x='smoking_status', y='stroke', estimator=lambda x: sum(x)/len(x))
        plt.title('Tỉ lệ đột quỵ theo tình trạng hút thuốc')
        plt.ylabel('Tỉ lệ đột quỵ')
        plt.ylim(0, 1)
        plt.xticks(rotation=15)
        img6_path = os.path.join(STATIC_FOLDER, 'plot_stroke_smoking.png')
        plt.savefig(img6_path)
        img_paths.append('plot_stroke_smoking.png')
        plt.close()

        return render_template("visualize.html", img_paths=img_paths)

    return render_template("visualize.html")

if __name__ == '__main__':
    app.run(debug=True)
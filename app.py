from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat model
model = joblib.load('diabetes_prediction_model.pkl')

# # Fungsi untuk membuat prediksi
# def make_prediction(input_data):
#     # Mengubah input_data menjadi DataFrame
#     input_df = pd.DataFrame([input_data])
#     # Membuat prediksi
#     prediction = model.predict(input_df)
#     return prediction[0]

# Fungsi untuk membuat prediksi
def make_prediction(input_data):
    # Mengubah input_data menjadi DataFrame
    input_df = pd.DataFrame([input_data])
    # Membuat prediksi
    prediction = model.predict(input_df)
    # Mengonversi prediksi menjadi tipe data int Python standar (jika diperlukan)
    prediction = int(prediction[0])
    return prediction


# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk membuat prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari request
    input_data = request.json
    # Membuat prediksi
    result = make_prediction(input_data)
    print(result)
    return jsonify({'prediction': result})
    # Merender ulang index.html dengan hasil
    # return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

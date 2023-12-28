# Kode untuk menghasilkan model Decision Tree Classifier dengan scikit-learn
# Menggunakan dataset diabetes yang telah dibersihkan

# Mengimpor pustaka yang diperlukan
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Memuat dataset yang telah dibersihkan
cleaned_file_path = 'cleaned_diabetes_prediction_dataset.csv'
cleaned_diabetes_data = pd.read_csv(cleaned_file_path)

# Mengatasi masalah nilai yang hilang
cleaned_diabetes_data['gender'].fillna(cleaned_diabetes_data['gender'].mode()[0], inplace=True)
cleaned_diabetes_data['smoking_history'].fillna(cleaned_diabetes_data['smoking_history'].median(), inplace=True)

# Memisahkan dataset menjadi fitur (X) dan target (y)
X = cleaned_diabetes_data.drop('diabetes', axis=1)
y = cleaned_diabetes_data['diabetes']

# Membagi dataset menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Menyimpan model yang telah dilatih
model_file_path = 'diabetes_prediction_model.pkl'
joblib.dump(model, model_file_path)

# Mengevaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy, model_file_path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data CSV dan mengambil kolom "Hours Studied" dan "Performance Index"
data = pd.read_csv('Student_Performance.csv')
X = data[['Hours Studied']]
y = data['Performance Index']

# Membuat model regresi linear
linear_model = LinearRegression()
linear_model.fit(X, y)

# Memprediksi menggunakan model regresi linear
y_predict_linear = linear_model.predict(X)

# Menghitung galat RMS untuk model regresi linear
rms_linear = np.sqrt(mean_squared_error(y, y_predict_linear))

# Plot grafik titik data dan hasil regresi linear
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Titik Data', s=1)
plt.plot(X, y_predict_linear, color='red', label= 'Hasil Regresi Linear')
plt.xlabel('Waktu Belajar (x)')
plt.ylabel('Nilai Ujian (y)')
plt.suptitle('Hasil Regresi Linear', fontsize=16)
plt.title(f'Galat RMS: {rms_linear}', fontsize=12, x=0.15)
plt.legend()
plt.show()

print(f'Galat RMS: {rms_linear}')
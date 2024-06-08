import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Membaca data CSV dan mengambil kolom "Hours Studied" dan "Performance Index"
data = pd.read_csv('Student_Performance.csv')
X = data['Hours Studied']
y = data['Performance Index']

# Urutkan data berdasarkan nilai X
sorted_indices = np.argsort(X)
X = X[sorted_indices]
y = y[sorted_indices]

# Definisikan model pangkat
def power_law(x, C, b):
    return C * np.power(x, b)

# Gunakan curve_fit untuk menemukan parameter a dan b
params, covariance_matrix = curve_fit(power_law, X, y)  # variabel "covariance_matrix" tidak digunakan
C, b = params

# Prediksi menggunakan model pangkat
y_predict = power_law(X, C, b)

# Menghitung galat RMS
rms_error = np.sqrt(mean_squared_error(y, y_predict))

# Plot hasil
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Titik Data', s=1)
plt.plot(X, y_predict, color='red', label= 'Hasil Regresi Pangkat Sederhana')
plt.xlabel('Waktu Belajar (x)')
plt.ylabel('Nilai Ujian (y)')
plt.suptitle('Hasil Regresi Pangkat Sederhana', fontsize=16)
plt.title(f'Galat RMS: {rms_error}', fontsize=12, x=0.18)
plt.legend()
plt.show()

print(f'Parameter C: {C}')
print(f'Parameter b: {b}')
print(f'Galat RMS:{rms_error}')

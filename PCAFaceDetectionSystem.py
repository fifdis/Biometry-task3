import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tkinter as tk


# загрузка изображений
def load_ORL_faces(data_folder):
    images = []
    for i in range(1, 41):
        for j in range(1, 11):
            img_path = os.path.join(data_folder, f"s{i}", f"{j}.pgm")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img.flatten())
    return np.array(images)

# PCA для метода с уменьшением изображения
def pca_image_reduction(X):
    # центрирование данных
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # вычисление матрицы ковариации
    cov_matrix = np.cov(X_centered.T)
    # вычисление собственных значений и собственных векторов
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    return cov_matrix, eigenvalues

# PCA с использованием матрицы Грамма-Шмидта
def pca_gram_schmidt(X):
    # центрирование данных
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    # вычисление матрицы Грамма-Шмидта
    gram_matrix = np.dot(X_centered, X_centered.T)
    q, _ = np.linalg.qr(gram_matrix)
    # вычисление собственных значений
    eigenvalues, _ = np.linalg.eigh(q)
    return gram_matrix, eigenvalues

# загрузка базы данных лиц ORL
data_folder = "ORLdataset"
images = load_ORL_faces(data_folder)

# вычисление матриц ковариации и собственных чисел для двух методов
cov_matrix_image, eigenvalues_image = pca_image_reduction(images)
cov_matrix_gram, eigenvalues_gram = pca_gram_schmidt(images)

# визуализация результатов
plt.figure(figsize=(15, 10))

# отображение первой матрицы ковариации
plt.subplot(2, 2, 1)
plt.imshow(cov_matrix_image, cmap='gray')
plt.title('Ковариационная матрица (уменьшение изображения)')

# отображение второй матрицы ковариации
plt.subplot(2, 2, 2)
plt.imshow(cov_matrix_gram, cmap='gray')
plt.title('Ковариационная матрица (преобразование Грама-Шмидта)')

# отображение графика собственных чисел для первого метода
plt.subplot(2, 2, 3)
plt.plot(eigenvalues_image)
plt.title('Собственные числа (уменьшение изображения)')

# отображение графика собственных чисел для второго метода
plt.subplot(2, 2, 4)
plt.plot(eigenvalues_gram)
plt.title('Собственные числа (преобразование Грама-Шмидта)')

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MultilayerPerceptron:
    def __init__(self, layers):
        # Inicializa la red neuronal con los tamaños de cada capa
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Calcula la derivada de la función sigmoide
        return x * (1 - x)

    def adelante_p(self, X):
        # Realiza la propagación hacia adelante
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.layers) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.z_values.append(z)
            self.activations.append(a)

    def atras_p(self, X, y, learning_rate):
        # Realiza la retropropagación para ajustar los pesos y sesgos
        errors = [y - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.layers) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        for i in range(len(self.layers) - 2, -1, -1):
            self.weights[i] += self.activations[i].T.dot(deltas[len(self.layers) - 2 - i]) * learning_rate
            self.biases[i] += np.sum(deltas[len(self.layers) - 2 - i], axis=0, keepdims=True) * learning_rate

    def entrenamiento(self, X, y, epochs, learning_rate):
        # Entrena la red durante un número específico de épocas
        for epoch in range(epochs):
            self.adelante_p(X)
            self.atras_p(X, y, learning_rate)

    def predict(self, X):
        # Realiza predicciones utilizando la red neuronal entrenada
        self.adelante_p(X)
        return np.round(self.activations[-1])

# Cargar los datos
data = pd.read_csv('concentlite.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Generar 1000 puntos de datos aleatorios para pruebas adicionales
np.random.seed(42)
X_extra_test = np.random.rand(1000, 2)
y_extra_test = np.random.randint(0, 2, size=(1000,))

# Visualizar los datos de prueba adicionales
plt.scatter(X_extra_test[y_extra_test==0, 0], X_extra_test[y_extra_test==0, 1], color='green', label='Datos de prueba extra clase 0', alpha=0.7)
plt.scatter(X_extra_test[y_extra_test==1, 0], X_extra_test[y_extra_test==1, 1], color='blue', label='Datos de prueba extra clase 1', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Datos de prueba extras:')
plt.legend()
plt.show()

# Dividir en conjunto de entrenamiento y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la arquitectura de la red
layers = [X.shape[1], 8, 1]

# Inicializar la red
mlp = MultilayerPerceptron(layers)

# Entrenar la red con más épocas y una tasa de aprendizaje más alta
mlp.entrenamiento(X_train, y_train.reshape(-1, 1), epochs=5000, learning_rate=0.2)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predictions = np.round(mlp.predict(X_test))

# Visualizar el resultado de la clasificación
plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], color='green', label='Clase 1 (Real)', alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions.flatten(), cmap='viridis', marker='x', label='Clase 2', linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificacion del perceptron multicapa')
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class RedNeuronal:
    def __init__(self, neuronas_capa):
        # Inicializa la red neuronal con el número de neuronas en cada capa
        self.neuronas_capa = neuronas_capa
        # Pesos iniciales aleatorios y sesgos a cero
        self.pesos = [np.random.randn(neuronas_capa[i], neuronas_capa[i + 1]) for i in range(len(neuronas_capa) - 1)]
        self.sesgo = [np.zeros((1, neuronas_capa[i + 1])) for i in range(len(neuronas_capa) - 1)]

    def sigmoide(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def derivada_sigmoide(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def feed_forward(self, X):
        # Propagación hacia adelante
        self.activaciones = [X]
        self.valores_z = []

        for i in range(len(self.neuronas_capa) - 1):
            # Producto punto y activación de la capa
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.sesgo[i]
            # Selección de la función de activación correspondiente
            if i == len(self.neuronas_capa) - 2:
                # Última capa, función softmax
                a = self.softmax(z)
            else:
                # Capas anteriores, función sigmoide
                a = self.sigmoide(z)
            # Guardar valores para la retropropagación
            self.valores_z.append(z)
            self.activaciones.append(a)

    def softmax(self, x):
        # Mejora la estabilidad numérica
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def retropropagacion(self, X, y, tasa_aprendizaje):
        # Ajuste de pesos y sesgos
        errores = [y - self.activaciones[-1]]
        deltas = [errores[-1]]

        for i in range(len(self.neuronas_capa) - 2, 0, -1):
            # Calcular errores y deltas para cada capa
            error = deltas[-1].dot(self.pesos[i].T)
            delta = error * self.derivada_sigmoide(self.activaciones[i])
            errores.append(error)
            deltas.append(delta)

        for i in range(len(self.neuronas_capa) - 2, -1, -1):
            # Actualizar pesos y sesgos utilizando errores y deltas
            self.pesos[i] += self.activaciones[i].T.dot(deltas[len(self.neuronas_capa) - 2 - i]) * tasa_aprendizaje
            self.sesgo[i] += np.sum(deltas[len(self.neuronas_capa) - 2 - i], axis=0, keepdims=True) * tasa_aprendizaje

    def entrenar_red(self, X, y, épocas, tasa_aprendizaje):
        # Entrenamiento de la red neuronal
        for época in range(épocas):
            # Propagación hacia adelante y retropropagación
            self.feed_forward(X)
            self.retropropagacion(X, y, tasa_aprendizaje)

    def predecir_clase(self, X):
        # Predicción utilizando la red neuronal entrenada
        self.feed_forward(X)
        return self.activaciones[-1]

    def evaluar_loo(self, X, y):
        # Evaluación utilizando Leave-One-Out
        loo = LeaveOneOut()
        precisiones = []

        for índice_entrenamiento, índice_prueba in loo.split(X):
            X_entrenamiento, X_prueba = X[índice_entrenamiento], X[índice_prueba]
            y_entrenamiento, y_prueba = y[índice_entrenamiento], y[índice_prueba]
            self.entrenar_red(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
            y_predicción_onehot = self.predecir_clase(X_prueba)
            y_predicción = np.argmax(y_predicción_onehot, axis=1)
            y_real = np.argmax(y_prueba, axis=1)
            precisión = accuracy_score(y_real, y_predicción)
            precisiones.append(precisión)

        return np.mean(precisiones), np.std(precisiones)

    def evaluar_lko(self, X, y, k):
        # Evaluación utilizando Leave-k-Out
        lko = KFold(n_splits=5)
        precisiones = []

        for índice_entrenamiento, índice_prueba in lko.split(X):
            X_entrenamiento, X_prueba = X[índice_entrenamiento], X[índice_prueba]
            y_entrenamiento, y_prueba = y[índice_entrenamiento], y[índice_prueba]
            self.entrenar_red(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
            y_predicción_onehot = self.predecir_clase(X_prueba)
            y_predicción = np.argmax(y_predicción_onehot, axis=1)
            y_real = np.argmax(y_prueba, axis=1)
            precisión = accuracy_score(y_real, y_predicción)
            precisiones.append(precisión)

        return np.mean(precisiones), np.std(precisiones)

# Cargar datos
datos = pd.read_csv('irisbin.csv', header=None)
# Separar características (X) y etiquetas (y)
X = datos.iloc[:, :-3].values
y = datos.iloc[:, -3:].values
# Escalar características
escalador = StandardScaler()
X = escalador.fit_transform(X)
# Definir arquitectura de la red
neuronas_capa = [X.shape[1], 8, 3]

# Inicializar la red
red_neuronal = RedNeuronal(neuronas_capa)
# Dividir conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
red_neuronal.entrenar_red(X_entrenamiento, y_entrenamiento, épocas=1000, tasa_aprendizaje=0.2)
# Hacer predicciones en el conjunto de prueba
predicciones = red_neuronal.predecir_clase(X_prueba)

# Evaluar la red
lko_promedio_precisión, lko_desviación_estándar = red_neuronal.evaluar_lko(X, y, k=5)
loo_promedio_precisión, loo_desviación_estándar = red_neuronal.evaluar_loo(X, y)
lko_error = 1 - lko_promedio_precisión
loo_error = 1 - loo_promedio_precisión

# Resultados
print("leave-k-out")
print("Error Esperado:", lko_error)
print("Promedio:", lko_promedio_precisión)
print("Desviacion Estandar:", lko_desviación_estándar)
print("leave-one-out")
print("Error Esperado:", loo_error)
print("Promedio:", loo_promedio_precisión)
print("Desviación Estandar:", loo_desviación_estándar)

# Predicciones y datos reales
print("Predicciones y Especies Reales:")
for i in range(len(predicciones)):
    especie_real = None
    if y_prueba[i][2] == 1:
        especie_real = 'Setosa'
    elif y_prueba[i][1] == 1:
        especie_real = 'Versicolor'
    elif y_prueba[i][0] == 1:
        especie_real = 'Virginica'
    
    especie_predicha = None
    if np.argmax(predicciones[i]) == 2:
        especie_predicha = 'Setosa'
    elif np.argmax(predicciones[i]) == 1:
        especie_predicha = 'Versicolor'
    elif np.argmax(predicciones[i]) == 0:
        especie_predicha = 'Virginica'

    print(f"Prediccion {i+1}: {especie_predicha}, Especie real = {especie_real}")

# Visualización
plt.scatter(X_prueba[y_prueba[:, 0] == 1, 0], X_prueba[y_prueba[:, 0] == 1, 1], color='#FF5733', label='Setosa', alpha=0.7)
plt.scatter(X_prueba[y_prueba[:, 1] == 1, 0], X_prueba[y_prueba[:, 1] == 1, 1], color='#33FF57', label='Versicolor', alpha=0.7)
plt.scatter(X_prueba[y_prueba[:, 2] == 1, 0], X_prueba[y_prueba[:, 2] == 1, 1], color='#5733FF', label='Virginica', alpha=0.7)

plt.xlabel('Pétalo')
plt.ylabel('Sépalo')
plt.title('Especies de Iris - Clasificación Red Neuronal')
plt.legend(loc='lower right', bbox_transform=plt.gcf().transFigure)
plt.show()

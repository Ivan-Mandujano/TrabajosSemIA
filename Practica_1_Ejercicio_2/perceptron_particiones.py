import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

#Función del perceptrón
def perceptron(entradas, pesos, sesgo):
    suma = np.dot(entradas, pesos) + sesgo
    #Función de activación
    if suma >= 0:
        return 1
    else:
        return 0

#Leer archivos de datos
    #spheres2d10.csv
    #spheres2d50.csv
    #spheres2d70.csv
archivo_conjunto_datos = 'spheres2d50.csv'
nombre_conjunto_datos = 'Conjunto de datos: spheres2d50'
datos = pd.read_csv(archivo_conjunto_datos)
entradas = datos.iloc[:, :-1].values
salidas = datos.iloc[:, -1].values

#Parámetros de entrenamiento
max_epocas = 100
tasa_aprendizaje = 0.1
criterio_convergencia = 0.01  #Alteraciones aleatorias < 5%

#Inicializar validación cruzada, 10 particiones
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

precision_promedio = 0.0

#Dividir entrenamiento (80%) y prueba (20%)

datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(entradas, salidas, test_size=0.2, random_state=42)

#Iterar en las particiones
for indice_particion, (indices_entrenamiento, indices_prueba) in enumerate(skf.split(entradas, salidas)):
    datos_entrenamiento, datos_prueba = entradas[indices_entrenamiento], entradas[indices_prueba]
    etiquetas_entrenamiento, etiquetas_prueba = salidas[indices_entrenamiento], salidas[indices_prueba]

    #Entrenamiento perceptron
    def entrenar_perceptron(entradas, salidas, tasa_aprendizaje, max_epocas, criterio_convergencia):
        num_entradas = entradas.shape[1]
        num_patrones = entradas.shape[0]

        pesos = np.random.rand(num_entradas)
        sesgo = np.random.rand()
        epocas = 0
        convergencia = False

        while epocas < max_epocas and not convergencia:
            convergencia = True
            for i in range(num_patrones):
                entrada = entradas[i]
                prediccion_salida = salidas[i]
                salida_recibida = np.dot(pesos, entrada) + sesgo
                error = prediccion_salida - salida_recibida

                if abs(error) > criterio_convergencia:
                    convergencia = False
                    pesos += tasa_aprendizaje * error * entrada
                    sesgo += tasa_aprendizaje * error
            epocas += 1
        return pesos, sesgo

    pesos_entrenados, sesgo_entrenado = entrenar_perceptron(datos_entrenamiento, etiquetas_entrenamiento, tasa_aprendizaje, max_epocas, criterio_convergencia)
    print(f"Perceptron entrenado exitosamente para {nombre_conjunto_datos} - Particion {indice_particion + 1}.")

    #Perceptrón entrenado en datos de prueba
    def probar_perceptron(entradas, pesos, sesgo):
        salida_recibida = np.dot(entradas, pesos) + sesgo
        return np.where(salida_recibida >= 0,1,0)

    predicciones_prueba = probar_perceptron(datos_prueba, pesos_entrenados, sesgo_entrenado)

    #Calcular la precisión
    precision = accuracy_score(etiquetas_prueba, predicciones_prueba)
    print(f"Precision del perceptron en datos de prueba para {nombre_conjunto_datos} - Particion {indice_particion + 1}: {precision * 100:.2f}%")

    #Particion actual + precisión promedio
    precision_promedio += precision

#Calcular precisión promedio
precision_promedio /= 10  # Dividir por el número de particiones (k-fold)

print(f"Precision promedio del perceptron en {nombre_conjunto_datos}: {precision_promedio * 100:.2f}%")

def plot_decision_boundary(ax, entradas, salidas, pesos, sesgo):
    # Scatter plot de patrones
    ax.scatter(entradas[:, 0], entradas[:, 1], entradas[:, 2], c=salidas, cmap=plt.cm.RdYlBu, s=100)

    # Generar una malla para trazar el plano de separación
    x_min, x_max = entradas[:, 0].min() - 1, entradas[:, 0].max() + 1
    y_min, y_max = entradas[:, 1].min() - 1, entradas[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Calcular valores de z utilizando el plano de separación
    z = (-pesos[0] * xx - pesos[1] * yy - sesgo) / pesos[2]

#Graficas
def grafico_3d(entradas, salidas, pesos, sesgo, titulo):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot del plano de separación
    plot_decision_boundary(ax, entradas, salidas, pesos, sesgo)

    ax.set_xlabel('Entrada X1')
    ax.set_ylabel('Entrada X2')
    ax.set_zlabel('Entrada X3')
    plt.title(titulo)
    plt.grid(True)
    plt.show()

#Graficar ejemplo de una de las particiones
grafico_3d(datos_entrenamiento, etiquetas_entrenamiento, pesos_entrenados, sesgo_entrenado, f'Patrones y Plano de Separación (3D) - {nombre_conjunto_datos} - Partición 1')
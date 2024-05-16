import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns

def zoo():
    #Cargar y dividir el dataset
    dataset = pd.read_csv('zoo_animals.csv')
    X = dataset.drop(['animal_name','type'],axis=1)
    y = dataset['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    return X_train, X_test, y_train, y_test



def regresion_logistica(X_train, X_test, y_train, y_test):
    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #Metricas utilizadas
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    Sensibilidad = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    matriz_confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()[:4]
    especificidad = tn / (tn + fp)

    #Matriz de confusión
    plt.figure(figsize=(8, 6))  # Cambiar el tamaño de la figura
    sns.set(font_scale=1.2)  # Aumentar el tamaño de la fuente
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), 
                linewidths=0.5, linecolor='black', cbar=True)

    plt.title('Matriz de Confusión RL', fontsize=16)
    plt.xlabel('Valor Predicho', fontsize=14)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mejor legibilidad
    plt.yticks(rotation=0)  # Asegurarse de que las etiquetas en el eje y no estén rotadas
    plt.show()
    
    print("\nRegresión Logística:\n")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {Sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    
    #Graficar metricas
    palette = sns.color_palette("husl", 5) 
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'Puntuación F1']
    valores_metricas = [exactitud, precision, Sensibilidad, especificidad, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(nombres_metricas, valores_metricas, color= palette)
    plt.title('Métricas para Regresión Logística')
    plt.xlabel('Métricas')
    plt.ylabel('Resultados')
    for i, valor in enumerate(valores_metricas):
        plt.text(i, valor + 0.01, f'{valor:.4f}', ha='center', va='bottom')
    plt.show()

##################################################################################################
def vecinos_mas_cercanos(X_train, X_test, y_train, y_test):
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    Sensibilidad = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    matriz_confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()[:4]
    especificidad = tn / (tn + fp)

    #Matriz de confusión
    plt.figure(figsize=(8, 6))  # Cambiar el tamaño de la figura
    sns.set(font_scale=1.2)  # Aumentar el tamaño de la fuente
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), 
                linewidths=0.5, linecolor='black', cbar=True)

    plt.title('Matriz de Confusión KV+C', fontsize=16)
    plt.xlabel('Valor Predicho', fontsize=14)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mejor legibilidad
    plt.yticks(rotation=0)  # Asegurarse de que las etiquetas en el eje y no estén rotadas
    plt.show()
    
    print("\nVecinos Más Cercanos:\n")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {Sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    
    #Graficar métricas
    palette = sns.color_palette("husl", 5) 
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'Puntuación F1']
    valores_metricas = [exactitud, precision, Sensibilidad, especificidad, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(nombres_metricas, valores_metricas, color= palette)
    plt.title('Métricas para Vecinos Más Cercanos')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    for i, valor in enumerate(valores_metricas):
        plt.text(i, valor + 0.01, f'{valor:.4f}', ha='center', va='bottom')
    plt.show()
    
#####################################################################################################
def maquina_vectores_soporte(X_train, X_test, y_train, y_test):
    modelo = SVC(C=1.0)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #Métricas utilizadas
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    Sensibilidad = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    matriz_confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()[:4]
    especificidad = tn / (tn + fp)
    
    #Matriz de confusión
    plt.figure(figsize=(8, 6))  # Cambiar el tamaño de la figura
    sns.set(font_scale=1.2)  # Aumentar el tamaño de la fuente
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), 
                linewidths=0.5, linecolor='black', cbar=True)

    plt.title('Matriz de Confusión MSV', fontsize=16)
    plt.xlabel('Valor Predicho', fontsize=14)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mejor legibilidad
    plt.yticks(rotation=0)  # Asegurarse de que las etiquetas en el eje y no estén rotadas
    plt.show()
    
    print("\nMáquina de Vectores de Soporte:\n")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {Sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Puntuación F1: {f1:.4f}")  

    #Graficar métricas
    palette = sns.color_palette("husl", 5) 
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'Puntuación F1']
    valores_metricas = [exactitud, precision, Sensibilidad, especificidad, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(nombres_metricas, valores_metricas, color= palette)
    plt.title('Métricas para Máquina de Vectores de Soporte')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    for i, valor in enumerate(valores_metricas):
        plt.text(i, valor + 0.01, f'{valor:.4f}', ha='center', va='bottom')
    plt.show()
    
################################################################################################
def naive_Bayes(X_train, X_test, y_train, y_test):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #Métricas utilizadas
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    Sensibilidad = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    matriz_confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()[:4]
    especificidad = tn / (tn + fp)

    #Matriz de confusión
    plt.figure(figsize=(8, 6))  # Cambiar el tamaño de la figura
    sns.set(font_scale=1.2)  # Aumentar el tamaño de la fuente
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), 
                linewidths=0.5, linecolor='black', cbar=True)

    plt.title('Matriz de Confusión NB', fontsize=16)
    plt.xlabel('Valor Predicho', fontsize=14)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mejor legibilidad
    plt.yticks(rotation=0)  # Asegurarse de que las etiquetas en el eje y no estén rotadas
    plt.show()
    
    print("\nNaive Bayes:\n")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {Sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    
    #Graficar métricas
    palette = sns.color_palette("husl", 5) 
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'Puntuación F1']
    valores_metricas = [exactitud, precision, Sensibilidad, especificidad, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(nombres_metricas, valores_metricas, color= palette)
    plt.title('Métricas para Bayes Ingenuo')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    for i, valor in enumerate(valores_metricas):
        plt.text(i, valor + 0.01, f'{valor:.4f}', ha='center', va='bottom')
    plt.show()
    
######################################################################################################################
def MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,50), max_iter=2000):
    modelo = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #Metricas utilizadas
    exactitud = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    Sensibilidad = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    matriz_confusion = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz_confusion.ravel()[:4]
    especificidad = tn / (tn + fp)

    #Matriz de confusión
    plt.figure(figsize=(8, 6))  # Cambiar el tamaño de la figura
    sns.set(font_scale=1.2)  # Aumentar el tamaño de la fuente
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), 
                linewidths=0.5, linecolor='black', cbar=True)

    plt.title('Matriz de Confusión MLP', fontsize=16)
    plt.xlabel('Valor Predicho', fontsize=14)
    plt.ylabel('Valor Real', fontsize=14)
    plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mejor legibilidad
    plt.yticks(rotation=0)  # Asegurarse de que las etiquetas en el eje y no estén rotadas
    plt.show()
    
    print("\nMLP:\n")
    print(f"Exactitud: {exactitud:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad: {Sensibilidad:.4f}")
    print(f"Especificidad: {especificidad:.4f}")
    print(f"Puntuación F1: {f1:.4f}")
    
    #Graficar métricas
    palette = sns.color_palette("husl", 5) 
    nombres_metricas = ['Exactitud', 'Precisión', 'Sensibilidad', 'Especificidad', 'Puntuación F1']
    valores_metricas = [exactitud, precision, Sensibilidad, especificidad, f1]
    plt.figure(figsize=(8, 6))
    plt.bar(nombres_metricas, valores_metricas, color= palette)
    plt.title('Métricas para MLP')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    for i, valor in enumerate(valores_metricas):
        plt.text(i, valor + 0.01, f'{valor:.4f}', ha='center', va='bottom')
    plt.show()

#Clasificadores
X_train, X_test, y_train, y_test = zoo()
regresion_logistica(X_train, X_test, y_train, y_test)
vecinos_mas_cercanos(X_train, X_test, y_train, y_test)
maquina_vectores_soporte(X_train, X_test, y_train, y_test)
naive_Bayes(X_train, X_test, y_train, y_test)
MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(16,16), max_iter=1000)
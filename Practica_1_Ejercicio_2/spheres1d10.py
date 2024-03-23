import pandas as pd

#Cargar el archivo CSV
datos = pd.read_csv('spheres1d10.csv')

#Definir las cinco particiones
num_particiones = 5

#Definir los porcentajes de datos para entrenamiento y prueba
porcentaje_entrenamiento = 0.8
porcentaje_prueba = 0.2

#Calcular los tamaños exactos de entrenamiento y prueba
tamaño_entrenamiento = int(len(datos) * porcentaje_entrenamiento)
tamaño_prueba = int(len(datos) * porcentaje_prueba)

#Hacer las particiones
for i in range(num_particiones):
    #Dividir entrenamiento y prueba de forma exacta
    datos_entrenamiento = datos.sample(n=tamaño_entrenamiento)
    datos_prueba = datos.sample(n=tamaño_prueba)
    
    #Imprimir particion
    print(f'Particion {i+1}: {len(datos_entrenamiento)+1} datos de entrenamiento, {len(datos_prueba)+1} datos de prueba')
    
    #Guardar entrenamiento y prueba en archivos separados
    datos_entrenamiento.to_csv(f'particion_entrenamiento_{i}.csv', index=False)
    datos_prueba.to_csv(f'particion_prueba_{i}.csv', index=False)
    
    #Combinar entrenamiento y prueba en un solo DataFrame
    datos_combinados = pd.concat([datos_entrenamiento, datos_prueba])
    
    #Guardar los datos combinados en un solo archivo por partición
    datos_combinados.to_csv(f'particion_combinada_{i}.csv', index=False)
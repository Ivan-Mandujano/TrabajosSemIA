import numpy as np
import matplotlib.pyplot as plt

def target_function(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

def gradient(x1, x2):
    return np.array([-2*x1*np.exp(-(x1**2 + 3*x2**2)),
                     -6*x2*np.exp(-(x1**2 + 3*x2**2))])

def update_parameters(x, learning_rate, gradient):
    return x - learning_rate * gradient

def optimize_function(learning_rate, epochs):
    initial_parameters = np.random.uniform(-1, 1, 2)
    error_history = []

    best_solution = {'x1': None, 'x2': None, 'error': float('inf')}

    for epoch in range(epochs):
        grad = gradient(initial_parameters[0], initial_parameters[1])
        initial_parameters = update_parameters(initial_parameters, learning_rate, grad)
        error = target_function(initial_parameters[0], initial_parameters[1])
        error_history.append(error)
        if error < best_solution['error']:
            best_solution['x1'] = initial_parameters[0]
            best_solution['x2'] = initial_parameters[1]
            best_solution['error'] = error

    print("Mejor solución encontrada por el algoritmo:")
    print(f"Valor X1: {best_solution['x1']}")
    print(f"Valor X2: {best_solution['x2']}")
    print(f"Error mínimo encontrado: {best_solution['error']}")

    # Graficar la convergencia del error
    plt.plot(range(epochs), error_history)
    plt.title('Convergencia del Error')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

# Ingresar datos desde la terminal
learning_rate = float(input("Ingrese el learning rate: "))
epochs = int(input("Ingrese la cantidad de épocas: "))

# Validación de límites
if not (0 < learning_rate < 1):
    print("El learning rate debe estar en el rango (0, 1)")
elif epochs <= 0:
    print("La cantidad de épocas debe ser un número positivo")
else:
    optimize_function(learning_rate, epochs)
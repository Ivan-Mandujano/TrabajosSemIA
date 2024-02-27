import numpy as np
import matplotlib.pyplot as plt

# Función del perceptrón
def custom_perceptron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    # Función de activación
    if weighted_sum >= 0:
        return 1
    else:
        return 0

# Leer el archivo csv
def read_patterns(file):
    data = np.genfromtxt(file, delimiter=',')
    inputs = data[:, :-1]
    outputs = data[:, -1]
    return inputs, outputs

# Entrenamiento del perceptrón
def train_perceptron(inputs, outputs, learning_rate, max_epochs, convergence_criteria):
    num_inputs = inputs.shape[1]
    num_patterns = inputs.shape[0]

    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    epochs = 0
    convergence = False

    while epochs < max_epochs and not convergence:
        convergence = True
        for i in range(num_patterns):
            input_pattern = inputs[i]
            desired_output = outputs[i]
            received_output = np.dot(weights, input_pattern) + bias
            error = desired_output - received_output

            if abs(error) > convergence_criteria:
                convergence = False
                weights += learning_rate * error * input_pattern
                bias += learning_rate * error
        epochs += 1
    return weights, bias

# Testear el perceptrón ya entrenado
def test_perceptron(inputs, weights, bias):
    received_output = np.dot(inputs, weights) + bias
    return np.sign(received_output)

# Calcular la precisión
def calculate_accuracy(actual_outputs, predicted_outputs):
    correct_predictions = np.sum(actual_outputs == predicted_outputs)
    total_predictions = len(actual_outputs)
    accuracy = correct_predictions / total_predictions
    return accuracy

def plot_graph(inputs, outputs, weights, bias):
    plt.figure(figsize=(8, 6))
    # Graficar patrones
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, s=100, cmap=plt.cm.coolwarm)

    # Graficar recta de separación
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = test_perceptron(np.c_[xx.ravel(), yy.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0])
    plt.title('Patrones y Línea de Separación')
    plt.xlabel('Entrada X1')
    plt.ylabel('Entrada X2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Lectura de patrones de entrenamiento y prueba desde archivos CSV
    training_file = 'OR_trn.csv'
    test_file = 'OR_tst.csv'
    # Patrones de entrenamiento
    training_inputs, training_outputs = read_patterns(training_file)
    # Patrones de prueba
    test_inputs, test_outputs = read_patterns(test_file)

    # Parámetros de entrenamiento
    max_epochs = 100
    learning_rate = 0.1
    convergence_criteria = 0.01  # Alteraciones aleatorias < 5%

    # Entrenamiento
    trained_weights, trained_bias = train_perceptron(training_inputs, training_outputs, learning_rate,
                                                    max_epochs, convergence_criteria)
    print("El perceptron ha concluido su entrenamiento")
    # Datos de prueba para probar el perceptrón
    test_outputs_predicted = test_perceptron(test_inputs, trained_weights, trained_bias)

    # Precisión
    accuracy = calculate_accuracy(test_outputs, test_outputs_predicted)
    print("Precision con datos de prueba:", accuracy)

    # Resultados
    print("Salidas en prueba:")
    print(test_outputs)
    print("Salidas predichas por el perceptron:")
    print(test_outputs_predicted)
    plot_graph(training_inputs, training_outputs, trained_weights, trained_bias)

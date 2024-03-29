from src.gradient.util import sigmoid, sigmoid_prime
import numpy as np

def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """

    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)


class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        activation_function_derivative - производная активационной функции нейрона
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1),
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """

        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)

    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        # Этот метод необходимо реализовать
        return input_matrix.dot(self.w)

    def activation(self, summatory_activation):
        """
        Вычисляет для каждого примера результат активационной функции,
        получив на вход вектор значений сумматорной функций
        summatory_activation - вектор размера (n, 1),
        где summatory_activation[i] - значение суммматорной функции для i-го примера.
        Возвращает вектор размера (n, 1), содержащий в i-й строке
        значение активационной функции для i-го примера.
        """
        # Этот метод необходимо реализовать
        return self.activation_function(summatory_activation)

    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция логистического нейрона.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
        (элементы вектора - float)
        """
        return self.activation(self.summatory(input_matrix))

    def J_quadratic(self, X, y):
        """
        Оценивает значение квадратичной целевой функции.
        Всё как в лекции, никаких хитростей.

        neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)

        Возвращает значение J (число)
        """

        assert y.shape[1] == 1, 'Incorrect y shape'

        return 0.5 * np.mean((self.vectorized_forward_pass(X) - y) ** 2)



    def compute_grad_analytically(self, X, y, J_prime=J_quadratic_derivative):
        """
        Аналитическая производная целевой функции
        neuron - объект класса Neuron
        X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
        y - правильные ответы для примеров из матрицы X
        J_prime - функция, считающая производные целевой функции по ответам

        Возвращает вектор размера (m, 1)
        """

        # Вычисляем активации
        # z - вектор результатов сумматорной функции нейрона на разных примерах

        z = self.summatory(X)
        y_hat = self.activation(z)
        # Вычисляем нужные нам частные производные
        dy_dyhat = J_prime(y, y_hat)
        dyhat_dz = self.activation_function_derivative(z)

        # осознайте эту строчку:
        dz_dw = X

        # а главное, эту:
        grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

        # можно было написать в два этапа. Осознайте, почему получается одно и то же
        # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
        # grad = np.sum(, axis=0)

        # Сделаем из горизонтального вектора вертикальный
        grad = grad.T

        return grad

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)

        learning_rate - константа скорости обучения
        batch_size - размер батча, на основании которого
        рассчитывается градиент и совершается один шаг алгоритма

        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.
        Вторым вариантом была бы проверка размера градиента, а не изменение функции,
        что будет работать лучше - неочевидно. В заданиях используйте первый подход.

        max_steps - критерий остановки номер два: если количество обновлений весов
        достигло max_steps, то алгоритм останавливается

        Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
        и 0, если второй (спуск не достиг минимума за отведённое время).
        """

        # Этот метод необходимо реализовать
        result = 0
        for _ in range(max_steps):
            random_idx = np.random.choice(range(len(X)), batch_size, replace=False)
            X_batched = X[random_idx]
            y_bathed = y[random_idx]
            result = self.update_mini_batch(X_batched, y_bathed, learning_rate, eps)
            if result == 1:
                break
        return result

    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        X - матрица размера (batch_size, m)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.

        Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
        и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
        иначе возвращаем 0.
        """
        # Этот метод необходимо реализовать

        # этот метод реализован в gradient.compute_grad_analytically, ну хоть разобрался
        # def J():
        #
        #     S = self.summatory(X)                      # без ф. активации w * X
        #     y_hat = self.vectorized_forward_pass(X)    # с применением ф.а. к w * X
        #     dJ_dy_hat = (y_hat - y) / len(y)                # 1 / n * (sigm(w * X)) - 1-я часть градиента
        #     dy_hat_dS = self.activation_function_derivative(S)
        #     dS_dw = X
        #     return ((dJ_dy_hat * dy_hat_dS).T).dot(dS_dw)
        # grad = J()

        grad = self.compute_grad_analytically(X, y)
        target_func = self.J_quadratic(X, y)
        # обновить веса
        self.w = self.w - (learning_rate * grad)
        # рассчитать новый градиент, не нужно, т.к. рассчитываем значение целевой ф-ции на новых весах
        # new_grad = compute_grad_analytically(self, X, y)
        target_func_new = self.J_quadratic(X, y)
        # если значение старой целевой ф-ции - значение новой целевой ф-ции < eps, возвращаем 1 иначе 0
        return int((target_func - target_func_new) < eps)

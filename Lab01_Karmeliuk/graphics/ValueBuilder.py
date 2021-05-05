import numpy as np
from scipy import interpolate


class ValueBuilder:
    @staticmethod
    def build_learn_x(l: int) -> list:
        result = list()
        for i in range(1, l + 1):
            x = 4 * ((i - 1) / (l - 1)) - 2
            result.append(x)
        return result

    @staticmethod
    def build_test_x(l: int) -> list:
        result = list()
        for i in range(1, l):
            x = 4 * ((i - 0.5) / (l - 1)) - 2
            result.append(x)
        return result

    @staticmethod
    def build_y(x_list) -> list:
        result = list()
        for x in x_list:
            result.append(ValueBuilder.function(x))
        return result

    @staticmethod
    def lagrange_polynomial(x_dots, y_dots, x):
        res = 0
        for i in range(len(x_dots)):
            p1 = p2 = 1
            for j in range(len(x_dots)):
                if i != j:
                    p1 *= (x - x_dots[j])
                    p2 *= (x_dots[i] - x_dots[j])
            res += y_dots[i] * p1 / p2
        return res

    @staticmethod
    def lagrange_test(x_dots, y_dots, x):
        res = 0
        for xi, yi in zip(x_dots, y_dots):
            res += yi * np.prod((x - x_dots[xi != x_dots]) / (xi - x_dots[x_dots != xi]))
        return res

    @staticmethod
    def lagrange_another(x_dots, y_dots, x):
        res = 0.0
        for xi, yi in zip(x_dots, y_dots):
            p = 1
            for xj in x_dots:
                if xi != xj:
                    p *= (x - xj) / (xi - xj)
            res += p * yi
        return res

    @staticmethod
    def function(x):
        return 1 / (1 + 25 * pow(x, 2))

    @staticmethod
    def least_squares_lagrange(x_dots, y_dots, test_x, test_y):
        res = 0
        for xi, yi in zip(test_x, test_y):
            res += (yi - ValueBuilder.lagrange_test(x_dots, y_dots, xi)) ** 2
        return res

    @staticmethod
    def least_squares_model(x_dots, y_dots, function):
        res = 0
        for xi, yi in zip(x_dots, y_dots):
            res += (yi - function(xi)) ** 2
        if res < 1E-17:
            return 0
        return res

    @staticmethod
    def build_polynom_model(degree, x_dots, y_dots):
        if degree > 1:
            return np.polyfit(x_dots, y_dots, degree)

        n = len(x_dots)
        x_sum = np.sum(x_dots)
        x_squared_sum = np.sum(np.power(x_dots, 2))
        y_sum = np.sum(y_dots)
        xy_sum = np.sum(np.multiply(x_dots, y_dots))

        matrix_left = np.array([[n, x_sum], [x_sum, x_squared_sum]])
        matrix_right = np.array([[y_sum], [xy_sum]])

        inverse_matrix_left = np.linalg.inv(matrix_left)
        result = np.multiply(inverse_matrix_left, matrix_right)

        return [result[1, 1], result[0, 0]]

    @staticmethod
    def build_spline(degree, x_dots, y_dots):
        if degree > 1:
            return interpolate.InterpolatedUnivariateSpline(x_dots, y_dots, k=degree)

        def build_spline_1_pow(x):
            for i, xi in enumerate(x_dots):
                if x > xi:
                    continue
                return (x - xi) / (x_dots[i - 1] - xi) * (y_dots[i - 1] - y_dots[i]) + y_dots[i]
            return 0

        return build_spline_1_pow

from random import random

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
    def build_y(x_list, chance_of_extremal, extremal_value) -> list:
        result = list()
        for x in x_list:
            random_num = random()
            value = ValueBuilder.function(x)
            if random_num > chance_of_extremal:
                # result.append(value + (random_num - 0.4) * 0.05)
                result.append(value)
            else:
                print("x anomaly ", x)
                if random() > 0.5:
                    result.append(value + extremal_value)
                else:
                    result.append(value - extremal_value)
        return result

    @staticmethod
    def function(x):
        return 1 / (1 + 25 * np.power(x, 2))

    @staticmethod
    def distance_flat(x1, x2):
        return np.abs(x1 - x2[0])

    @staticmethod
    def distance_to_dots(x, x_dots, distance_method):
        result = list()
        for xi in x_dots:
            result.append([xi, distance_method(x, xi)])
        return result

    @staticmethod
    def nadaraya_watson_function(distance_dots, kernel, h):
        top = 0
        bottom = 0
        for xi in distance_dots:
            # xi = [ [x,y,coeff], distance ]
            # without LOWESS coeff is always 1
            r = xi[1] / h
            coef = kernel(r) * xi[0][2]
            top += coef * xi[0][1]
            bottom += coef
        if bottom > 1E-15:
            return top / bottom
        return 0

    @staticmethod
    def nadaraya_watson_function_sliding(distance_dots, kernel, k_neighbour):
        top = 0
        bottom = 0
        for xi in distance_dots:
            r = xi[1] / distance_dots[k_neighbour][1]
            coef = kernel(r) * xi[0][2]
            top += coef * xi[0][1]
            bottom += coef
        if bottom > 1E-15:
            return top / bottom
        return 0

    @staticmethod
    def kernel_rect(val: float) -> float:
        absolute = np.abs(val)
        if absolute <= 1:
            return 0.5
        else:
            return 0

    @staticmethod
    def kernel_triangle(val: float) -> float:
        absolute = np.abs(val)
        if absolute <= 1:
            return 1 - absolute
        else:
            return 0

    @staticmethod
    def kernel_square(val: float) -> float:
        absolute = np.abs(val)
        if absolute <= 1:
            return 1 - np.power(absolute, 2)
        else:
            return 0

    @staticmethod
    def kernel_gauss(val: float) -> float:
        return np.exp(-2 * np.power(val, 2))

    @staticmethod
    def get_random_x(x1, x2):
        return np.random.uniform(x1, x2)

    @staticmethod
    def euclidian_distance(dot1, dot2):
        return np.sqrt(np.power(dot1[0] - dot2[0], 2) + np.power(dot1[1] - dot2[1], 2))

    @staticmethod
    def classify_miss_by_best(learning_dots, distance_method, kernel, h, sliding=False):
        res = 0
        for index, learning_dot in enumerate(learning_dots):
            testing_data = np.concatenate([learning_dots[:index], learning_dots[index + 1:]])
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            if sliding:
                test_value = ValueBuilder.nadaraya_watson_function_sliding(list(sorted(distances, key=lambda l: l[1])),
                                                                           kernel, h)
            else:
                test_value = ValueBuilder.nadaraya_watson_function(distances, kernel, h)
            res += (learning_dot[1] - test_value) ** 2
        if res < 1E-17:
            return 0
        return res

    @staticmethod
    def learn_best_h(learning_dots, distance_method, kernel):
        result = dict()
        h_range = np.arange(0.001, 2.0, 0.001)
        for index, h in enumerate(h_range):
            result[index] = ValueBuilder.classify_miss_by_best(learning_dots, distance_method, kernel, h)
            print(h, result[index])
        print(result)
        return h_range[min(result, key=result.get)]

    @staticmethod
    def learn_best_neighbours(learning_dots, distance_method, kernel):
        result = dict()
        for k in range(1, 20, 1):
            result[k] = ValueBuilder.classify_miss_by_best(learning_dots,
                                                           distance_method,
                                                           kernel,
                                                           k, sliding=True)
            print(k, result[k])
        print(result)
        return min(result, key=lambda key: result[key])

    @staticmethod
    def select_best_kernel(learning_dots, distance_method, kernels, k, sliding=False):
        result = dict()
        for index, kernel in enumerate(kernels):
            print(kernel)
            if sliding:
                result[index] = ValueBuilder.classify_miss_by_best(learning_dots,
                                                                   distance_method,
                                                                   kernel,
                                                                   k, sliding=True)
            else:
                result[index] = ValueBuilder.classify_miss_by_best(learning_dots,
                                                                   distance_method,
                                                                   kernel,
                                                                   k)
        print(result)
        return min(result, key=result.get)

    # kvarticheskoe = (1-r^2)^2
    @staticmethod
    def lowess_kernel(val):
        # if val <= 1:
        #     return np.power(1 - np.power(val, 2),2)
        # else:
        #     return 0
        return np.exp(-2 * np.power(val, 2))

    @staticmethod
    def lowess(learning_dots, distance_method, kernel, h, sliding=False):
        # learning dots are pairs (x,y)
        coefs = np.full(len(learning_dots), fill_value=1)
        dots_with_coefs = np.concatenate((learning_dots, coefs[:, None]), axis=1)
        for _ in range(4):
            for index, learning_dot in enumerate(dots_with_coefs):
                testing_data = np.concatenate([dots_with_coefs[:index], dots_with_coefs[index + 1:]])
                distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
                if sliding:
                    test_value = ValueBuilder.nadaraya_watson_function_sliding(
                        list(sorted(distances, key=lambda l: l[1])), kernel, h)
                else:
                    test_value = ValueBuilder.nadaraya_watson_function(distances, kernel, h)
                learning_dot[2] = ValueBuilder.lowess_kernel(np.abs(learning_dot[1] - test_value))
        return dots_with_coefs

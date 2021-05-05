import math

import numpy as np
import time


class ValueBuilder:

    @staticmethod
    def spread_segment_equally(x1, x2, n):
        return np.linspace(x1, x2, num=n)

    @staticmethod
    def classify_dots(ab_segment, cd_segment, learning_dots):
        result = list()
        for learning_dot in learning_dots:
            for i1, x1 in enumerate(ab_segment):
                if learning_dot[0] < x1:
                    for i2, x2 in enumerate(cd_segment):
                        if learning_dot[1] < x2:
                            result.append([learning_dot, i1 + (i2 - 1) * (len(ab_segment) - 1)])
                            break
                    break
        return result

    @staticmethod
    def euclidian_distance(dot1, dot2):
        return np.sqrt(np.power(dot1[0] - dot2[0], 2) + np.power(dot1[1] - dot2[1], 2))

    @staticmethod
    def manhattan_distance(dot1, dot2):
        return np.absolute(dot1[0] - dot2[0]) + np.absolute(dot1[1] - dot2[1])

    @staticmethod
    def distance_to_dots(dot, learning_dots, distance_method):
        result = list()
        for learning_dot in learning_dots:
            result.append([learning_dot[0], learning_dot[1], distance_method(dot, learning_dot[0])])
        return list(sorted(result, key=lambda x: x[2]))

    @staticmethod
    def get_random_point(a, b, c, d):
        return [np.random.uniform(a, b), np.random.uniform(c, d)]

    @staticmethod
    def select_best_kernel(best_dots, learning_dots, distance_method, kernels, h):
        result = dict()
        for index, kernel in enumerate(kernels):
            print(kernel)
            result[index] = ValueBuilder.classify_miss_by_best(best_dots, learning_dots, distance_method, kernel, h)
        print(result)
        return max(result, key=result.get)

    @staticmethod
    def select_best_kernel_sliding(best_dots, learning_dots, distance_method, kernels, k):
        result = dict()
        for index, kernel in enumerate(kernels):
            print(kernel)
            result[index] = ValueBuilder.classify_miss_by_best_sliding(best_dots,
                                                                       learning_dots,
                                                                       distance_method,
                                                                       kernel,
                                                                       k)
        print(result)
        return max(result, key=result.get)

    @staticmethod
    def select_best_h(best_dots, learning_dots, distance_method, kernel):
        result = dict()
        h_range = np.arange(0.01, 0.12, 0.01)
        for index, h in enumerate(h_range):
            result[index] = ValueBuilder.classify_miss_by_best(best_dots, learning_dots, distance_method, kernel, h)
            print(h, result[index])
        print(result)
        return h_range[max(result, key=result.get)]

    @staticmethod
    def select_best_k_sliding(best_dots, learning_dots, distance_method, kernel):
        result = dict()
        for k in range(1, 20, 1):
            result[k] = ValueBuilder.classify_miss_by_best_sliding(best_dots,
                                                                   learning_dots,
                                                                   distance_method,
                                                                   kernel,
                                                                   k)
            print(k, result[k])
        print(result)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def classify_dot_by_parzen_window(neighbours, h, kernel) -> int:
        result = dict()
        for index, neighbour in enumerate(neighbours):
            r = neighbour[2] / h
            try:
                result[neighbour[1]] += kernel(r)
            except KeyError:
                result[neighbour[1]] = kernel(r)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def classify_dot_by_parzen_window_sliding(neighbours, k, kernel) -> int:
        result = dict()
        for index, neighbour in enumerate(neighbours):
            r = neighbour[2] / neighbours[k][2]
            try:
                result[neighbour[1]] += kernel(r)
            except KeyError:
                result[neighbour[1]] = kernel(r)
        return max(result, key=lambda key: result[key])

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
    def sort_into_classes(classified_dots):
        # coords , class
        result = dict()
        for dot in classified_dots:
            try:
                result[dot[1]].append(dot[0])
            except KeyError:
                result[dot[1]] = list()
                result[dot[1]].append(dot[0])
        return result

    @staticmethod
    def sum_distance_to_dots(dot, learning_dots, distance_method):
        result = 0
        for learning_dot in learning_dots:
            result += distance_method(dot, learning_dot)
        return result

    @staticmethod
    def get_sorted_by_sum_distance(sorted_dots_by_class, distance_method):
        # class : coords
        result = dict()
        for dot_class in sorted_dots_by_class:
            for index, dot in enumerate(sorted_dots_by_class[dot_class]):
                testing_data = sorted_dots_by_class[dot_class][:index] + sorted_dots_by_class[dot_class][index + 1:]
                distances = ValueBuilder.sum_distance_to_dots(dot, testing_data, distance_method)
                try:
                    result[dot_class].append([dot, distances])
                except KeyError:
                    result[dot_class] = list()
                    result[dot_class].append([dot, distances])
        return ValueBuilder.sort_by_sum(result)

    @staticmethod
    def sort_by_sum(sorted_dots_by_class):
        for dot_class in sorted_dots_by_class:
            sorted_dots_by_class[dot_class] = sorted(sorted_dots_by_class[dot_class], key=lambda x: x[1])
        return sorted_dots_by_class

    @staticmethod
    def get_best_dots(dots_by_class, percentage):
        result = dict()
        for dot_class in dots_by_class:
            # print(len(dots_by_class[dot_class]),percentage)
            amount = int(len(dots_by_class[dot_class]) * percentage)
            if amount < 1:
                amount = 1
            result[dot_class] = dots_by_class[dot_class][:amount]
        return result

    @staticmethod
    def convert_to_list(best_dots_class):
        result = list()
        for dot_class in best_dots_class:
            for dot in best_dots_class[dot_class]:
                result.append(dot[0])
        return result

    @staticmethod
    def convert_to_list_with_class(best_dots_class):
        result = list()
        for dot_class in best_dots_class:
            for dot in best_dots_class[dot_class]:
                result.append([dot[0], dot_class])
        return result

    @staticmethod
    def classify_miss_by_best(best_dots, learning_dots, distance_method, kernel, h):
        result = 0
        for index, learning_dot in enumerate(learning_dots):
            testing_data = [x for x in best_dots if not ValueBuilder.are_equal(x, learning_dot)]
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            test_class = ValueBuilder.classify_dot_by_parzen_window(distances, h, kernel)
            if test_class == learning_dot[1]:
                result += 1
            else:
                continue

        return result

    @staticmethod
    def classify_miss_by_best_sliding(best_dots, learning_dots, distance_method, kernel, k):
        result = 0
        for index, learning_dot in enumerate(learning_dots):
            testing_data = [x for x in best_dots if not ValueBuilder.are_equal(x, learning_dot)]
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            test_class = ValueBuilder.classify_dot_by_parzen_window_sliding(distances, k, kernel)
            if test_class == learning_dot[1]:
                result += 1
            else:
                continue
        return result

    @staticmethod
    def are_equal(dot, other):
        return dot[0][0] == other[0][0] and dot[0][1] == other[0][1]

    @staticmethod
    def dot_in(dot, other_dots):
        for other_dot in other_dots:
            if dot[0] == other_dot[0] and dot[1] == other_dot[1]:
                return True
        return False

    @staticmethod
    def find_best_percentage(main_dots_classified, distance_method, kernel, h):
        dots_by_class = ValueBuilder.sort_into_classes(main_dots_classified)
        sorted_by_distance = ValueBuilder.get_sorted_by_sum_distance(dots_by_class, distance_method)
        result = dict()
        percentage_range = np.arange(0.01, 1.01, 0.01)
        for percentage in percentage_range:
            dots_by_class = ValueBuilder.get_best_dots(sorted_by_distance, percentage)
            best_dots = ValueBuilder.convert_to_list_with_class(dots_by_class)
            result[percentage] = ValueBuilder.classify_miss_by_best(best_dots, main_dots_classified, distance_method,
                                                                    kernel, h)
            print(percentage, result[percentage])
        print(result)
        return max(result, key=result.get)

    @staticmethod
    def find_best_percentage_sliding(main_dots_classified, distance_method, kernel, k):
        dots_by_class = ValueBuilder.sort_into_classes(main_dots_classified)
        sorted_by_distance = ValueBuilder.get_sorted_by_sum_distance(dots_by_class, distance_method)
        result = dict()
        percentage_range = np.arange(0.01, 1.01, 0.01)
        for percentage in percentage_range:
            dots_by_class = ValueBuilder.get_best_dots(sorted_by_distance, percentage)
            best_dots = ValueBuilder.convert_to_list_with_class(dots_by_class)
            result[percentage] = ValueBuilder.classify_miss_by_best_sliding(best_dots, main_dots_classified,
                                                                            distance_method,
                                                                            kernel, k)
            print(percentage, result[percentage])
        print(result)
        return max(result, key=result.get)

    # K nearest neighbours
    @staticmethod
    def no_weigh(i, k):
        return 1

    @staticmethod
    def linear_weight(i, k):
        return (k + 1 - i) / k

    @staticmethod
    def exponential_weight(i, k):
        # return np.power(1 / np.exp(1),i)
        return 1 / np.exp(i)

    @staticmethod
    def classify_by_k_neighbours(neighbours, weigh_function):
        result = dict()
        k = len(neighbours)
        for index, neighbour in enumerate(neighbours):
            try:
                result[neighbour[1]] += weigh_function(index, k)
            except KeyError:
                result[neighbour[1]] = weigh_function(index, k)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def select_right_k(best_dots, learning_dots, distance_method, weigh_function):
        result = dict()
        for k in range(1, 20, 1):
            result[k] = ValueBuilder.classify_miss_by_k(best_dots, learning_dots, distance_method, weigh_function, k)
            print(k, result[k])
        print(result)
        return max(result, key=lambda key: result[key])

    @staticmethod
    def classify_miss_by_k(best_dots, learning_dots, distance_method, weigh_function, k):
        result = 0
        # print(f"{distance_method} - {k} - {weigh_function}")
        for index, learning_dot in enumerate(learning_dots):
            testing_data = [x for x in best_dots if not ValueBuilder.are_equal(x, learning_dot)]
            distances = ValueBuilder.distance_to_dots(learning_dot[0], testing_data, distance_method)
            test_class = ValueBuilder.classify_by_k_neighbours(distances[:k], weigh_function)
            if test_class == learning_dot[1]:
                result += 1
            else:
                continue
        return result

    @staticmethod
    def select_best_weigh(best_dots, learning_dots, distance_method, weighs, k):
        result = dict()
        for index, weigh in enumerate(weighs):
            print(weigh)
            result[index] = ValueBuilder.classify_miss_by_k(best_dots, learning_dots, distance_method, weigh, k)
        print(result)
        return max(result, key=result.get)

    @staticmethod
    def find_best_percentage_k(main_dots_classified, distance_method, weigh_function, k):
        dots_by_class = ValueBuilder.sort_into_classes(main_dots_classified)
        sorted_by_distance = ValueBuilder.get_sorted_by_sum_distance(dots_by_class, distance_method)
        result = dict()
        percentage_range = np.arange(0.01, 1.01, 0.01)
        for percentage in percentage_range:
            dots_by_class = ValueBuilder.get_best_dots(sorted_by_distance, percentage)
            best_dots = ValueBuilder.convert_to_list_with_class(dots_by_class)
            result[percentage] = ValueBuilder.classify_miss_by_k(best_dots, main_dots_classified, distance_method,
                                                                 weigh_function, k)
            print(percentage, result[percentage])
        print(result)
        return max(result, key=result.get)

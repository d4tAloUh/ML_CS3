from datetime import datetime
from scipy.interpolate import lagrange
from graphics.ValueBuilder import ValueBuilder
import numpy as np


class GraphicBuilder:
    def __init__(self, sliding=False):
        self.sliding = sliding
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.MIN_X = -2
        self.MAX_X = 2

        self.GRAPH_MAX_X = 2.1
        self.GRAPH_MIN_X = -2.1

        self.GRAPH_MAX_Y = 1.5
        self.GRAPH_MIN_Y = -0.5

        self.GRAPHIC_STEP = 0.01

        self.DOT_LEARNING_SIZE = 26
        self.DOT_LEARNING_COLOR = 'lime'

        self.DOT_TESTING_SIZE = 6
        self.DOT_TESTING_COLOR = 'black'

        self.LINE_WIDTH = 1
        self.LINE_COLOR = 'black'

        self.MAX_LEARNING_DOTS_AMOUNT = 700
        self.MIN_LEARNING_DOTS_AMOUNT = 2

        self.LEARNING_DOTS_AMOUNT = 50

        self.SLIDER_STEP = 1

        # Main functions
        self.DISTANCE_METHOD = ValueBuilder.distance_flat
        self.KERNEL_FUNCTION = ValueBuilder.kernel_rect
        self.kernels = [ValueBuilder.kernel_rect, ValueBuilder.kernel_triangle, ValueBuilder.kernel_square,
                        ValueBuilder.kernel_gauss]

        # Nadaraya-Watson parameters
        self.h = 0.1
        if self.sliding:
            self.neighbour_sliding = 3
            self.MIN_SLIDING_NEIGHBOUR_NUM = 1
            self.MAX_SLIDING_NEIGHBOUR_NUM = 30

        self.EXTREMAL_CHANCE = 0.06
        self.EXTREMAL_VALUE = 2
        self.MIN_H = 0.001
        self.MAX_H = 2.0
        self.MAX_TEST_DOTS_AMOUNT = 700
        self.MIN_TEST_DOTS_AMOUNT = 2
        self.TEST_DOTS_AMOUNT = 120

        self.generate_dots()

        self.generate_random_x()
        self.get_values_for_x_dots()

        self.graphic_dots_x = np.arange(self.MIN_X, self.MAX_X + self.GRAPHIC_STEP, self.GRAPHIC_STEP)
        self.graphic_dots_y = self.build_graphic_dots_sin()
        print(self.learning_dots)



    def build_graphic_dots_sin(self):
        result = list()
        for x in self.graphic_dots_x:
            result.append(ValueBuilder.function(x))
        return np.array(result, float)

    def generate_dots(self):
        self.LEARNING_DOTS_X = np.array(ValueBuilder.build_learn_x(self.LEARNING_DOTS_AMOUNT), float)
        self.LEARNING_DOTS_Y = np.array(ValueBuilder.build_y(self.LEARNING_DOTS_X, self.EXTREMAL_CHANCE,self.EXTREMAL_VALUE), float)
        self.learning_dots = np.column_stack((self.LEARNING_DOTS_X, self.LEARNING_DOTS_Y))
        self.build_coeffs()

    def build_coeffs(self):
        if self.sliding:
            self.learning_dots = ValueBuilder.lowess(self.learning_dots, self.DISTANCE_METHOD, self.KERNEL_FUNCTION,
                                                     self.neighbour_sliding,sliding=True)
        else:
            self.learning_dots = ValueBuilder.lowess(self.learning_dots, self.DISTANCE_METHOD, self.KERNEL_FUNCTION, self.h)
        print(self.learning_dots)

    def generate_random_x(self):
        self.TEST_DOTS_X = list()
        for index in range(0, self.TEST_DOTS_AMOUNT):
            self.TEST_DOTS_X.append(ValueBuilder.get_random_x(self.MIN_X, self.MAX_X))

    def get_values_for_x_dots(self):
        self.TEST_DOTS_Y = list()
        for x_dot in self.TEST_DOTS_X:

            dot_distances = ValueBuilder.distance_to_dots(x_dot, self.learning_dots,
                                                          self.DISTANCE_METHOD)
            if self.sliding:
                self.TEST_DOTS_Y.append(
                    ValueBuilder.nadaraya_watson_function_sliding(list(sorted(dot_distances, key=lambda l: l[1])),
                                                                  self.KERNEL_FUNCTION,
                                                                  self.neighbour_sliding))
            else:
                self.TEST_DOTS_Y.append(ValueBuilder.nadaraya_watson_function(dot_distances, self.KERNEL_FUNCTION,
                                                                              self.h))
        self.TEST_DOTS = np.column_stack((self.TEST_DOTS_X, self.TEST_DOTS_Y))

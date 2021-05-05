from datetime import datetime
from scipy.interpolate import lagrange
from graphics.ValueBuilder import ValueBuilder
import numpy as np


class GraphicBuilder:
    def __init__(self):
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.MIN_X = -2
        self.MAX_X = 2

        self.GRAPH_MAX_X = 2.1
        self.GRAPH_MIN_X = -2.1

        self.GRAPH_MAX_Y = 5.0
        self.GRAPH_MIN_Y = -5.0

        self.GRAPHIC_STEP = 0.01

        self.DOT_LEARNING_SIZE = 20
        self.DOT_LEARNING_COLOR = 'red'
        self.DOT_TESTING_SIZE = 25
        self.DOT_TESTING_COLOR = 'darkblue'

        self.LINE_WIDTH = 1
        self.LINE_COLOR = 'black'

        self.MAX_DOTS_AMOUNT = 165
        self.MIN_DOTS_AMOUNT = 2
        self.SLIDER_STEP = 1
        self.DOTS_AMOUNT = 5

        self.VALUE_BUILDER = ValueBuilder()

        self.TEST_DOTS_X = np.array(ValueBuilder.build_test_x(self.DOTS_AMOUNT), float)
        self.TEST_DOTS_Y = np.array(ValueBuilder.build_y(self.TEST_DOTS_X), float)

        self.LEARN_DOTS_X = np.array(self.VALUE_BUILDER.build_learn_x(self.DOTS_AMOUNT), float)
        self.LEARN_DOTS_Y = np.array(self.VALUE_BUILDER.build_y(self.LEARN_DOTS_X), float)

        self.graphic_dots_x = np.arange(self.MIN_X, self.MAX_X + self.GRAPHIC_STEP, self.GRAPHIC_STEP)

        self.model = np.poly1d(ValueBuilder.build_polynom_model(1, self.LEARN_DOTS_X,
                                                                self.LEARN_DOTS_Y))
        self.sin_dots = self.build_graphic_dots_sin()

        self.spline = ValueBuilder.build_spline(1, self.LEARN_DOTS_X, self.LEARN_DOTS_Y)

    def build_graphic_dots(self, dots_x, dots_y):
        result = list()
        now = datetime.now()
        for x in self.graphic_dots_x:
            result.append(self.VALUE_BUILDER.lagrange_test(dots_x, dots_y, x))
        end = datetime.now()
        print("Time for calculating lagrange values: ", end - now)
        return np.array(result, float)

    def build_model_dots(self):
        result = list()
        for x in self.graphic_dots_x:
            result.append(self.model(x))
        return np.array(result, float)

    def build_spline_dots(self):
        result = list()
        for x in self.graphic_dots_x:
            result.append(self.spline(x))
        return np.array(result, float)

    def build_graphic_dots_sin(self):
        result = list()
        for x in self.graphic_dots_x:
            result.append(self.VALUE_BUILDER.function(x))
        return np.array(result, float)
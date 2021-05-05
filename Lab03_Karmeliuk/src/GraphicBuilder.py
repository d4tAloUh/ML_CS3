import numpy as np
from src.ValueBuilder import ValueBuilder


class GraphicBuilder:
    def __init__(self, sliding=False, k_nearest=False):
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.DOT_LEARNING_SIZE = 4
        self.DOT_LEARNING_COLOR = 'red'
        self.DOT_TESTING_SIZE = 20
        self.DOT_TESTING_COLOR = 'darkblue'
        self.DOT_NOT_BEST_SIZE = 3
        self.DOT_NOT_BEST_COLOR = 'darkmagenta'

        self.LINE_WIDTH = 0.5
        self.LINE_COLOR = 'black'
        self.SELECTED_WIDTH = 2.0

        self.MIN_CLASSES = 1
        self.MAX_CLASSES = 15

        self.MIN_H = 0.01
        self.MAX_H = 2

        self.MIN_K = 1
        self.MAX_K = 20
        self.K = 5
        # x1
        self.a = 14.0
        self.b = 16.0

        # x2
        self.c = 2.0
        self.d = 4.0

        # n - x1 classes, m - x2 classes
        self.n = 3
        self.m = 3
        self.sliding = sliding
        self.k_nearest = k_nearest

        # amount of Learning Dots
        self.l = 500
        self.MIN_LEARNING_DOTS = 10
        self.MAX_LEARNING_DOTS = 2000

        # h for parzen window
        if not sliding:
            self.h = 0.1

        self.PERCENTAGE = 1.0
        self.PERCENTAGE_DOT_AMOUNT = self.l
        self.KERNEL_ALGORITHM = ValueBuilder.kernel_rect
        self.DISTANCE_ALGORITHM = ValueBuilder.euclidian_distance

        self.KERNEL_ALGORITHMS = [ValueBuilder.kernel_rect, ValueBuilder.kernel_triangle, ValueBuilder.kernel_square,
                                  ValueBuilder.kernel_gauss]
        # building points for segments equally
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)

        self.generate_dots_and_clasify()
        self.get_best_dots()

        self.generate_and_clasify_dot()
        self.build_difference()
        self.horizontal_not_best = list(map(lambda x: x[0], self.not_best_dots))
        self.vertical_not_best = list(map(lambda x: x[1], self.not_best_dots))

    #     K NEAREST
        self.K_NEAREST = 4
        self.WEIGHT_ALGORITHM = ValueBuilder.no_weigh
        self.WEIGHTS = [ValueBuilder.no_weigh,ValueBuilder.linear_weight,ValueBuilder.exponential_weight]


    def generate_dots_and_clasify(self):
        # generating learning dots
        learning_dots_horizontal = np.random.uniform(self.a, self.b, size=self.l)
        learning_dots_vertical = np.random.uniform(self.c, self.d, size=self.l)
        self.main_dots = np.column_stack((learning_dots_horizontal, learning_dots_vertical))
        self.main_dots = list(sorted(self.main_dots, key=lambda x: (x[0], x[1])))

        # classifying dots
        self.main_classified = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.main_dots)

    def get_best_dots(self):
        # Selecting best dots
        dots_by_class = ValueBuilder.sort_into_classes(self.main_classified)
        sorted_by_distance = ValueBuilder.get_sorted_by_sum_distance(dots_by_class, self.DISTANCE_ALGORITHM)
        self.best_dots = ValueBuilder.get_best_dots(sorted_by_distance, self.PERCENTAGE)
        self.classified_dots = ValueBuilder.convert_to_list_with_class(self.best_dots)
        self.learning_dots = ValueBuilder.convert_to_list(self.best_dots)
        self.learning_dots_horizontal = list(map(lambda x: x[0], self.learning_dots))
        self.learning_dots_vertical = list(map(lambda x: x[1], self.learning_dots))
        self.PERCENTAGE_DOT_AMOUNT = len(self.learning_dots)

    def generate_ab_and_classify(self):
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.main_classified = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)
        self.get_best_dots()

    def generate_cd_and_classify(self):
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)
        self.main_classified = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)
        self.get_best_dots()

    def generate_and_clasify_dot(self):
        self.random_point = ValueBuilder.get_random_point(self.a, self.b, self.c, self.d)
        # Generating random point
        self.classify_dot()

    def classify_dot(self):
        # array(coords,class,distance)
        self.distances = ValueBuilder.distance_to_dots(self.random_point, self.classified_dots,
                                                       self.DISTANCE_ALGORITHM)
        if not self.sliding:
            self.point_class = ValueBuilder.classify_dot_by_parzen_window(self.distances, self.h, self.KERNEL_ALGORITHM)
        elif self.k_nearest:
            self.point_class = ValueBuilder.classify_by_k_neighbours(self.distances[:self.K_NEAREST],
                                                                     self.WEIGHT_ALGORITHM)
        else:
            self.h = self.distances[self.K - 1][2]
            self.point_class = ValueBuilder.classify_dot_by_parzen_window_sliding(self.distances, self.K,
                                                                                  self.KERNEL_ALGORITHM)

    def build_difference(self):
        if self.PERCENTAGE == 1.0:
            self.not_best_dots = []
        else:
            self.not_best_dots = [x for x in self.main_dots if not ValueBuilder.dot_in(x, self.learning_dots)]

    def get_hor_ver_class(self):
        h = 0
        while self.point_class - h * self.n > 0:
            h += 1
        h = h - 1 if h > 0 else 0
        v = self.point_class - h * self.n - 1
        return h, v

    def box_hor_ver(self):
        h, v = self.get_hor_ver_class()
        hor_res = [h, h + 1]
        ver_res = [v, v + 1]

        return hor_res, ver_res

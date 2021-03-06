import numpy as np
from src.ValueBuilder import ValueBuilder


class GraphicBuilder:
    def __init__(self):
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.DOT_LEARNING_SIZE = 1
        self.DOT_LEARNING_COLOR = 'red'
        self.DOT_TESTING_SIZE = 25
        self.DOT_TESTING_COLOR = 'darkblue'

        self.LINE_WIDTH = 0.5
        self.LINE_COLOR = 'black'
        self.SELECTED_WIDTH = 2.0

        self.MIN_CLASSES = 1
        self.MAX_CLASSES = 15

        self.MIN_K = 1
        self.MAX_K = 20
        # x1
        self.a = 14.0
        self.b = 16.0

        # x2
        self.c = 2.0
        self.d = 4.0

        # n - x1 classes, m - x2 classes
        self.n = 5
        self.m = 10

        # amount of Learning Dots
        self.l = 500
        self.MIN_LEARNING_DOTS = 10
        self.MAX_LEARNING_DOTS = 2000
        # amount of nearest members
        self.k = 4
        self.DISTANCE_ALGORITHM = ValueBuilder.euclidian_distance
        self.WEIGHT_ALGORITHM = ValueBuilder.no_weigh

        # building points for segments equally
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)

        self.generate_dots_and_clasify()

        self.generate_and_clasify_dot()

    def generate_dots_and_clasify(self):
        # generating learning dots
        self.learning_dots_horizontal = np.random.uniform(self.a, self.b, size=self.l)
        self.learning_dots_vertical = np.random.uniform(self.c, self.d, size=self.l)
        self.learning_dots = np.column_stack((self.learning_dots_horizontal, self.learning_dots_vertical))
        self.learning_dots = list(sorted(self.learning_dots, key=lambda x: (x[0], x[1])))
        # classifying dots
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_ab_and_classify(self):
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_cd_and_classify(self):
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_and_clasify_dot(self):
        # Generating random point
        self.random_point = ValueBuilder.get_random_point(self.a, self.b, self.c, self.d)
        self.classify_dot()

    def classify_dot(self):
        # array(coords,class,distance)
        self.distances = ValueBuilder.distance_to_dots(self.random_point, self.classified_dots,
                                                       self.DISTANCE_ALGORITHM)

        self.point_class = ValueBuilder.classify_by_k_neighbours(self.distances[:self.k], self.WEIGHT_ALGORITHM)

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

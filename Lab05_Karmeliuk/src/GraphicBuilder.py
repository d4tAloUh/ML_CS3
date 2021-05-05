from collections import deque

import numpy as np
from src.models import Leaf
from src.ValueBuilder import ValueBuilder


class GraphicBuilder:
    def __init__(self, tree=False):
        self.use_tree = tree
        self.WINDOW_WIDTH = 30 / 2.54
        self.WINDOW_HEIGHT = 13 / 2.54

        self.DOT_LEARNING_SIZE = 2
        self.DOT_LEARNING_COLOR = 'red'

        self.DOT_TESTING_SIZE = 5
        self.DOT_TESTING_COLOR = 'lime'

        self.RANDOM_DOT_SIZE = 12
        self.RANDOM_DOT_COLOR = 'darkblue'

        self.LINE_WIDTH = 0.5
        self.LINE_COLOR = 'black'
        self.SELECTED_WIDTH = 2.0

        self.MIN_CLASSES = 1
        self.MAX_CLASSES = 15

        # x1
        self.a = 14.0
        self.b = 16.0

        # x2
        self.c = 2.0
        self.d = 4.0

        # n - x1 classes, m - x2 classes
        self.n = 3
        self.m = 4

        # amount of Learning Dots
        self.l = 500
        self.MIN_LEARNING_DOTS = 10
        self.MAX_LEARNING_DOTS = 2000

        # building points for segments equally
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)

        self.selected_class = 1
        self.generate_dots_and_clasify()
        self.random_dot = ValueBuilder.get_random_point(self.a, self.b, self.c, self.d)

        self.eps_vert = 0
        self.eps_hor = 0

        self.zones = {}
        self.list = []
        self.tree = None
        self.classification_method = self.apply_tree

    def generate_random_dot(self):
        self.random_dot = ValueBuilder.get_random_point(self.a, self.b, self.c, self.d)

    def generate_dots_and_clasify(self):
        # generating learning dots
        self.learning_dots_horizontal = np.random.uniform(self.a, self.b, size=self.l)
        self.learning_dots_vertical = np.random.uniform(self.c, self.d, size=self.l)
        self.learning_dots = np.column_stack((self.learning_dots_horizontal, self.learning_dots_vertical))

        # classifying dots
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)
        if self.use_tree:
            self.testing_dots = self.classified_dots[:int(len(self.classified_dots) * 0.3)]
            self.classified_dots = self.classified_dots[int(len(self.classified_dots) * 0.3):]

    def generate_ab_and_classify(self):
        self.ab_segment = ValueBuilder.spread_segment_equally(self.a, self.b, self.n + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def generate_cd_and_classify(self):
        self.cd_segment = ValueBuilder.spread_segment_equally(self.c, self.d, self.m + 1)
        self.classified_dots = ValueBuilder.classify_dots(self.ab_segment, self.cd_segment, self.learning_dots)

    def get_ab_cd_points(self, class_num):
        index_2, index_1 = ValueBuilder.convert_class_to_indexes(class_num, len(self.ab_segment) - 1)
        width = self.ab_segment[index_1 + 1] - self.ab_segment[index_1]
        height = self.cd_segment[index_2 + 1] - self.cd_segment[index_2]
        return (self.ab_segment[index_1] + self.eps_hor * width,
                self.cd_segment[index_2] + self.eps_vert * height), width, height

    def calculate_informativeness(self):
        Pc, Nc = ValueBuilder.get_pn_by_class(self.classified_dots, self.selected_class)
        heuristic1, heuristic2 = ValueBuilder.get_informativeness(self.ab_segment, self.cd_segment,
                                                                  self.selected_class, self.classified_dots,
                                                                  ValueBuilder.heuristic_informativeness,
                                                                  eps_vert=self.eps_vert, eps_hor=self.eps_hor)

        statistic = ValueBuilder.get_informativeness(self.ab_segment, self.cd_segment, self.selected_class,
                                                     self.classified_dots,
                                                     ValueBuilder.statistic_informativeness,
                                                     eps_vert=self.eps_vert, eps_hor=self.eps_hor)

        entropy = ValueBuilder.get_informativeness(self.ab_segment, self.cd_segment, self.selected_class,
                                                   self.classified_dots, ValueBuilder.entropy_informativeness,
                                                   eps_vert=self.eps_vert, eps_hor=self.eps_hor)
        return f"{heuristic1},  {heuristic2}", statistic, entropy, \
               f"{1 / ((Pc + Nc) * np.log(2)) * statistic} = {entropy}"

    def get_eps(self, percentage: float):
        return (self.b - self.a) * percentage

    def build_zones(self):
        print(self.ab_segment)
        print(self.cd_segment)
        for class_num in range(1, self.n * self.m + 1):
            zones_ab = ValueBuilder.union_zones(self.classified_dots, class_num, ValueBuilder.rule_ab,
                                                ValueBuilder.entropy_informativeness, var_index=0)
            zones_cd = ValueBuilder.union_zones(self.classified_dots, class_num, ValueBuilder.rule_cd,
                                                ValueBuilder.entropy_informativeness, var_index=1)
            rules = ValueBuilder.transform_to_rules(zones_ab, zones_cd)
            self.zones[class_num] = rules
            print("Class", class_num, "zones:")
            print("x:", zones_ab)
            print("y:", zones_cd)
            print()

    def build_list(self):
        self.list = ValueBuilder.build_solution_list(self.classified_dots, self.n * self.m + 1, zones=self.zones)
        for class_num in self.list:
            print(class_num, "Class")
            print("Rules:", self.list[class_num])

    def build_tree(self):
        predicates = []
        for class_num in range(1, self.n * self.m + 1):
            if len(self.zones) < self.n * self.m:
                zones_ab = ValueBuilder.union_zones(self.classified_dots, class_num, ValueBuilder.rule_ab,
                                                    ValueBuilder.entropy_informativeness, var_index=0)
                zones_cd = ValueBuilder.union_zones(self.classified_dots, class_num, ValueBuilder.rule_cd,
                                                    ValueBuilder.entropy_informativeness, var_index=1)
                rules = ValueBuilder.transform_to_rules(zones_ab, zones_cd)
                self.zones[class_num] = rules
            else:
                rules = self.zones[class_num]
            predicates.extend(rules)
        self.tree = ValueBuilder.build_tree(self.classified_dots, predicates)
        self.print_tree(self.tree)
        self.pprint_tree(self.tree, has_right=False)

    def print_tree(self, tree):
        vertex_list = deque([(tree, 0)])
        while vertex_list:
            vertex, index = vertex_list.popleft()
            if isinstance(vertex, Leaf):
                print(index, ": Class", vertex.class_num)
            else:
                print(index, ": rule", vertex.rule, f"false {2 * index + 1}", f"true {2 * index + 2}")
                vertex_list.append((vertex.left, 2 * index + 1))
                vertex_list.append((vertex.right, 2 * index + 2))

    def pprint_tree(self, node, padding="", pointer="", has_right=True):
        if isinstance(node, Leaf):
            print(f"{padding}{pointer}Class:{int(node.class_num)}")
        else:
            print(f"{padding}{pointer}{node.rule}")
            if has_right:
                padding += "│  "
            else:
                padding += "   "
            pointer_right = "└──"

            pointer_left = "├──"
            self.pprint_tree(node.left, padding, pointer_left)
            self.pprint_tree(node.right, padding, pointer_right, False)

    def apply_tree(self, dot):
        if self.tree is None:
            self.build_tree()
        vertex = self.tree
        while not isinstance(vertex, Leaf):
            if vertex.rule.apply(dot):
                vertex = vertex.right
            else:
                vertex = vertex.left
        return int(vertex.class_num)

    def apply_list(self, dot):
        if len(self.list) == 0:
            self.build_list()
        for class_num in self.list:
            if all(map(lambda x: x.apply(dot), self.list[class_num])):
                return int(class_num)
        print("No result in list:", dot)
        return None

    def calculate_hit(self):
        hit = 0
        for dot in self.testing_dots:
            if dot[2] == self.classification_method(dot):
                hit += 1
        return hit

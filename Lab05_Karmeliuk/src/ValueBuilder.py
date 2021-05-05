import itertools
import math
import random

import numpy as np

from math import comb

from src.models import Rule, Leaf, Tree


class ValueBuilder:

    @staticmethod
    def spread_segment_equally(x1, x2, n):
        return np.linspace(x1, x2, num=n)

    @staticmethod
    def get_random_point(a, b, c, d):
        return [np.random.uniform(a, b), np.random.uniform(c, d)]

    @staticmethod
    def classify_dots(ab_segment, cd_segment, learning_dots):
        result = list()
        for learning_dot in learning_dots:

            for i1, x1 in enumerate(ab_segment):
                if learning_dot[0] < x1:
                    for i2, x2 in enumerate(cd_segment):
                        if learning_dot[1] < x2:
                            result.append(
                                np.array([learning_dot[0], learning_dot[1], i1 + (i2 - 1) * (len(ab_segment) - 1)]))
                            break
                    break
        return result

    @staticmethod
    def convert_class_to_indexes(class_num: int, x1_class_amount):
        h = 0
        while class_num - h * x1_class_amount > 0:
            h += 1
        h = h - 1 if h > 0 else 0
        v = class_num - h * x1_class_amount - 1
        return h, v

    @staticmethod
    def rule_ab(a, b, dot):
        if a <= dot[0] <= b:
            return 1
        return 0

    @staticmethod
    def rule_cd(c, d, dot):
        if c <= dot[1] <= d:
            return 1
        return 0

    @staticmethod
    def rule_with_ab_cd(a, b, c, d, dot):
        if ValueBuilder.rule_ab(a, b, dot) == 1:
            return ValueBuilder.rule_cd(c, d, dot)
        return 0

    @staticmethod
    def union_zones(classified_dots, class_num, rule, informativeness, number_of_zones=2, var_index=0):
        # [x,y,class]
        zones = []
        sorted_dots_by_x = list(sorted(classified_dots, key=lambda x: x[var_index]))

        # Create zones
        for index in range(0, len(sorted_dots_by_x) - 1):
            if sorted_dots_by_x[index][var_index] != sorted_dots_by_x[index + 1][var_index] and \
                    sorted_dots_by_x[index][2] == class_num and \
                    sorted_dots_by_x[index + 1][2] != class_num:
                zones.append((sorted_dots_by_x[index][var_index] + sorted_dots_by_x[index + 1][var_index]) * 1.0 / 2)

        # Union zones algorithm
        while len(zones) > number_of_zones + 1:
            increase = []
            for index in range(0, len(zones) - 1):
                left, right = ValueBuilder.get_left_and_right_zone_range(zones, index)
                union_informativeness = ValueBuilder.informativeness_for_zones(left, right, class_num,
                                                                               classified_dots,
                                                                               rule,
                                                                               informativeness)
                left_inf = ValueBuilder.informativeness_for_zones(left, zones[index], class_num, classified_dots,
                                                                  rule,
                                                                  informativeness)
                center_inf = ValueBuilder.informativeness_for_zones(zones[index], zones[index + 1], class_num,
                                                                    classified_dots,
                                                                    rule,
                                                                    informativeness)
                right_inf = ValueBuilder.informativeness_for_zones(zones[index + 1], right, class_num,
                                                                   classified_dots,
                                                                   rule,
                                                                   informativeness)
                increase.append(union_informativeness - max(left_inf, center_inf, right_inf))
            maximum = max(increase)
            index = increase.index(maximum)
            del zones[index:index + 2]
        return zones

    @staticmethod
    def informativeness_for_zones(val1, val2, class_num, classified_dots, rule, informativeness):
        pc = 0
        nc = 0
        for dot in classified_dots:
            rule_result = rule(val1, val2, dot)

            # If rule worked
            if rule_result == 1:
                # true classification
                if dot[2] == class_num:
                    pc += 1
                # false classification
                elif dot[2] != class_num:
                    nc += 1
            # if rule did not work just continue
        Pc, Nc = ValueBuilder.get_pn_by_class(classified_dots, class_num)
        return informativeness(pc, nc, Pc, Nc)

    @staticmethod
    def get_left_and_right_zone_range(zones, index):
        try:
            left = zones[index - 1]
        except IndexError:
            left = -math.inf
        try:
            right = zones[index + 2]
        except IndexError:
            right = math.inf
        return left, right

    @staticmethod
    def gradient_algorithm(rules, classified_dots, class_num, tmax, d, eps, informativeness):
        algo_informativeness = -math.inf
        best_combination = []
        result_t = 0
        phi_t = []
        Pc, Nc = ValueBuilder.get_pn_by_class(classified_dots, class_num)
        for t in range(tmax + 1):
            # print("================================================================================")
            combinations = ValueBuilder.mutate_combination(best_combination, rules)
            best_informativeness = -math.inf
            best_index = 0
            if algo_informativeness > 0.1:
                break
            for index, combination in enumerate(combinations):
                pc, nc = ValueBuilder.informativeness_of_DNF(combination, classified_dots, class_num)
                if pc + nc > 0 and nc / (pc + nc) < eps:
                    comb_informativeness = informativeness(pc, nc, Pc, Nc)
                    if comb_informativeness > best_informativeness:
                        best_informativeness = comb_informativeness
                        best_index = index
            if best_informativeness > algo_informativeness:
                algo_informativeness = best_informativeness
                result_t = t
                phi_t = combinations[best_index]
            best_combination = combinations[best_index]
            if t - result_t > d:
                break
        return ValueBuilder.short_combination(phi_t), algo_informativeness

    @staticmethod
    def short_combination(combination):
        result = []
        for rule in combination:
            other_rules = list(filter(lambda x: x.char == rule.char and x.sign == rule.sign, combination))
            if len(other_rules) > 1:
                if rule.sign:
                    best = min(other_rules, key=lambda x: x.val)
                else:
                    best = max(other_rules, key=lambda x: x.val)
            else:
                best = other_rules[0]
            if best not in result:
                result.append(best)
        return result

    @staticmethod
    def mutate_combination(combination, rules, amount=30):
        result = []
        for _ in range(amount):
            result_comb = []
            # for current_rule in combination:
            #     if random.random() < 0.8:
            #         result_comb.append(current_rule)
            for rule in rules:
                #     if rule not in result_comb and random.random() < 0.3:
                #         result_comb.append(rule)
                # if rule not in result_comb:
                #     if random.random() < 0.7:
                #         result_comb.append(rule)
                # if len(result_comb) > 5 and random.random() < 0.8:
                #     index_rule = result_comb.index(random.choice(result_comb))
                #     result_comb = result_comb[:index_rule] + result_comb[index_rule + 1:]
                #
                # if len(result_comb) < 1:
                #     result_comb.append(random.choice(rules))

                if rule not in result_comb:
                    rand = random.random()
                    if rand > 0.2:
                        result_comb.append(rule)
                    elif len(result_comb) > 6:
                        index_rule = result_comb.index(random.choice(result_comb))
                        result_comb = result_comb[:index_rule] + result_comb[index_rule + 1:]
                    # elif len(result_comb) > 3:
                    #     index_rule = result_comb.index(random.choice(result_comb))
                    #     result_comb = result_comb[:index_rule] + result_comb[index_rule + 1:]
                    #     result_comb.append(rule)

            if len(combination) < 1:
                result_comb.append(random.choice(rules))

            # else:
            #     index_rule = combination.index(random.choice(combination))
            #     result_comb = list(combination)
            #     if len(result_comb) > 7:
            #         result_comb = result_comb[:index_rule] + result_comb[index_rule + 1:]
            #     random_rule = random.choice(rules)
            #     while random_rule in result_comb:
            #         random_rule = random.choice(rules)
            #     result_comb.append(random_rule)

            result.append(result_comb)
        return result

    @staticmethod
    def informativeness_of_DNF(combination, classified_dots, class_num):
        pc = 0
        nc = 0
        for dot in classified_dots:
            if all(map(lambda x: x.apply(dot), combination)):
                # true classification
                if dot[2] == class_num:
                    pc += 1
                # false classification
                elif dot[2] != class_num:
                    nc += 1
            # if rule did not work just continue
        return pc, nc

    @staticmethod
    def transform_to_rules(ab, cd):
        result = []
        for index, ab_lim in enumerate(ab):
            # first should have values only on right side
            if index == 0:
                result.append(Rule(ab_lim))
            # last should only have values on left side
            elif index == len(ab) - 1:
                result.append(Rule(ab_lim, sign=False))
            else:
                result.append(Rule(ab_lim))
                result.append(Rule(ab_lim, sign=False))
        for index, cd_lim in enumerate(cd):
            # first should have values only on right side
            if index == 0:
                result.append(Rule(cd_lim, char="y"))
            # last should only have values on left side
            elif index == len(cd) - 1:
                result.append(Rule(cd_lim, sign=False, char="y"))
            else:
                result.append(Rule(cd_lim, char="y"))
                result.append(Rule(cd_lim, sign=False, char="y"))
        return result

    @staticmethod
    def apply_rule_to_dots(a, b, c, d, class_num, classified_dots, rule):
        # [x,y,class]
        pc = 0
        nc = 0
        for dot in classified_dots:
            if rule == ValueBuilder.rule_with_ab_cd:
                rule_result = rule(a, b, c, d, dot)
            elif rule == ValueBuilder.rule_ab:
                rule_result = rule(a, b, dot)
            else:
                rule_result = rule(c, d, dot)

            # If rule worked
            if rule_result == 1:
                # true classification
                if dot[2] == class_num:
                    pc += 1
                # false classification
                elif dot[2] != class_num:
                    nc += 1
            # if rule did not work just continue
        return pc, nc

    @staticmethod
    def get_informativeness(ab_segment, cd_segment, class_num, classified_dots, informativeness, eps_vert=0, eps_hor=0):
        index_2, index_1 = ValueBuilder.convert_class_to_indexes(class_num, len(ab_segment) - 1)
        pc, nc = ValueBuilder.apply_rule_to_dots(ab_segment[index_1] + eps_hor,
                                                 ab_segment[index_1 + 1] + eps_hor,
                                                 cd_segment[index_2] + eps_vert,
                                                 cd_segment[index_2 + 1] + eps_vert, class_num,
                                                 classified_dots, ValueBuilder.rule_with_ab_cd)
        Pc, Nc = ValueBuilder.get_pn_by_class(classified_dots, class_num)
        return informativeness(pc, nc, Pc, Nc)

    @staticmethod
    def get_pn_by_class(classified_dots, class_num):
        Pn = len(list(filter(lambda x: x[2] == class_num, classified_dots)))
        Nc = len(classified_dots) - Pn
        return Pn, Nc

    @staticmethod
    def heuristic_informativeness(pc: int, nc: int, Pc: int, Nc: int) -> (float, float):
        l = Pc + Nc
        return nc * 1.0 / (pc + nc + 0.00000001), pc * 1.0 / (l + 0.00000001)

    @staticmethod
    def statistic_informativeness(pc: int, nc: int, Pc: int, Nc: int) -> float:
        result = ValueBuilder.my_comb(Pc, pc) * ValueBuilder.my_comb(Nc, nc) / (
                ValueBuilder.my_comb(Pc + Nc, nc + pc) + 0.00000001)
        return -np.log(result)

    @staticmethod
    def my_comb(C, c):
        res = comb(C, c)
        if res == 0:
            return 1
        return res

    @staticmethod
    def entropy_informativeness(pc: int, nc: int, Pc: int, Nc: int) -> float:
        l = Pc + Nc
        l2 = pc + nc
        l3 = l - l2
        entopy_information = l2 / (l + 0.00000001) * ValueBuilder.entropy(pc / (l2 + 0.00000001)) + l3 / (
                l + 0.00000001) * ValueBuilder.entropy(
            (Pc - pc) / (l3 + 0.00000001))
        return ValueBuilder.entropy(Pc * 1.0 / l) - entopy_information

    @staticmethod
    def entropy(q: float) -> float:
        q = q + 0.00000001 if q < 0.00000001 else q
        return -q * np.log2(q) - (1 - q) * np.log2(1 - q)

    @staticmethod
    def build_solution_list(classified_dots, max_class, zones=None):
        solution_list = {}
        copy_dots = classified_dots
        for class_num in range(1, max_class):
            if zones is None:
                zones_ab = ValueBuilder.union_zones(copy_dots, class_num, ValueBuilder.rule_ab,
                                                    ValueBuilder.entropy_informativeness, var_index=0)
                zones_cd = ValueBuilder.union_zones(copy_dots, class_num, ValueBuilder.rule_cd,
                                                    ValueBuilder.entropy_informativeness, var_index=1)
                rules = ValueBuilder.transform_to_rules(zones_ab, zones_cd)
            else:
                rules = zones[class_num]
            # print(class_num, "rules:", rules)
            best_conj, inform = ValueBuilder.gradient_algorithm(rules, copy_dots, class_num, 100, 50, 0.2,
                                                                ValueBuilder.entropy_informativeness)
            solution_list[class_num] = best_conj
            copy_dots = ValueBuilder.filter_out_classified(copy_dots, best_conj)
        return solution_list

    @staticmethod
    def filter_out_classified(dots, rule):
        result = []
        for dot in dots:
            if not all(map(lambda x: x.apply(dot), rule)):
                result.append(dot)
        return result

    @staticmethod
    def build_tree(dots, predicates):
        classes = list(set(map(lambda x: x[2], dots)))
        if len(classes) == 1:
            return Leaf(class_num=classes[0])

        best_predicate = ValueBuilder.find_best_predicate(dots, predicates)

        left_tree, right_tree = ValueBuilder.split_dots(dots, best_predicate)
        if len(left_tree) == 0 or len(right_tree) == 0:
            vertex = Leaf(ValueBuilder.get_max_class(dots))
        else:
            vertex = Tree(best_predicate)
            vertex.left = ValueBuilder.build_tree(left_tree, predicates)
            vertex.right = ValueBuilder.build_tree(right_tree, predicates)
        return vertex

    @staticmethod
    def find_best_predicate(dots, predicates):
        best_info = -math.inf
        best_predicate = None
        for predicate in predicates:
            pred_info = ValueBuilder.statistical_full_informativeness(dots, predicate)
            if pred_info > best_info:
                best_info = pred_info
                best_predicate = predicate
        return best_predicate

    @staticmethod
    def split_dots(dots, predicate):
        first = []
        second = []
        for dot in dots:
            if predicate.apply(dot):
                second.append(dot)
            else:
                first.append(dot)
        return first, second

    @staticmethod
    def get_max_class(dots):
        classes = {}
        for dot in dots:
            try:
                classes[dot[2]] += 1
            except KeyError:
                classes[dot[2]] = 1
        return max(classes, key=lambda key: classes[key])

    @staticmethod
    def statistical_full_informativeness(dots, predicate):
        classes_num = list(set(map(lambda x: x[2], dots)))
        top = 1
        dots_classified = 0
        for class_num in classes_num:
            P = 0
            pc = 0
            for dot in dots:
                if dot[2] == class_num:
                    P += 1
                    if predicate.apply(dot):
                        pc += 1
            dots_classified += pc
            top *= ValueBuilder.my_comb(P, pc)
        return -np.log(top * 1.0 / ValueBuilder.my_comb(len(dots), dots_classified))

    @staticmethod
    def convert_to_list(dots):
        result = []
        for dot in dots:
            result.append([dot[0], dot[1]])
        return result

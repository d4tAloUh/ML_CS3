class Rule:
    def __init__(self, val, sign=True, char="x"):
        self.val = val
        # if true means >=, else <=
        self.sign = sign
        self.char = char

    def apply(self, dot):
        # if sign is >=, then val should be on the right side of self.val
        val = dot[0] if self.char == "x" else dot[1]
        if self.sign:
            return self.val <= val
        return self.val >= val

    def apply_with_sign(self, dot, sign):
        val = dot[0] if self.char == "x" else dot[1]
        if sign:
            return self.val <= val
        return self.val >= val

    def __str__(self):
        return f"{self.val} < {self.char}" if self.sign else f"{self.char} < {self.val}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.char == other.char and self.sign == other.sign and abs(self.val - other.val) < 0.000000001


class Tree:
    def __init__(self, rule):
        self.rule = rule
        self.left = None
        self.right = None


class Leaf:
    def __init__(self, class_num):
        self.class_num = class_num

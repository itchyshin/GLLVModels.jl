"""Compare JSON values without treating booleans as numbers."""
import math


def json_value_equal(left, right):
    """Compare finite JSON values recursively, preserving JSON value types."""
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left is right
    if type(left) in (int, float) or type(right) in (int, float):
        if type(left) not in (int, float) or type(right) not in (int, float):
            return False
        try:
            return math.isfinite(left) and math.isfinite(right) and left == right
        except OverflowError:
            return False
    if left is None or right is None:
        return left is None and right is None
    if type(left) is str or type(right) is str:
        return type(left) is str and type(right) is str and left == right
    if type(left) is list and type(right) is list:
        return len(left) == len(right) and all(json_value_equal(a, b) for a, b in zip(left, right))
    if type(left) is dict and type(right) is dict:
        return (all(type(k) is str for k in left) and all(type(k) is str for k in right)
                and left.keys() == right.keys()
                and all(json_value_equal(left[k], right[k]) for k in left))
    return False

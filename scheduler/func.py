def decay_func_1(steps: int)->float:
    b = steps + 1
    b = b ** 0.25
    return 1/b

def decay_func_2(steps: int)->float:
    b = steps + 1
    b = b ** 0.50
    return 1/b

def decay_func_3(steps: int)->float:
    b = steps + 1
    b = b ** 0.75
    return 1/b

def decay_func_4(steps: int)->float:
    b = steps + 1
    b = b ** 1
    return 1/b
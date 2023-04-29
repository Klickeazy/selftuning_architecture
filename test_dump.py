import greedy_architecture_combined as gac
import shelve
import numpy as np
from copy import deepcopy as dc
import control

class A:
    def __init__(self):
        self.variable1 = self.B()
        self.variable2 = 3

    class B:
        def __init__(self):
            self.valueB = 10

    class C:
        def __init__(self):
            self.valueC = 5
            self.variable3 += 1
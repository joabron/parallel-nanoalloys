#!/usr/bin/env python

import bigfloat as bf
import numpy as np

class bf_array:

    def __init__(self, array, context=bf.precision(256)) -> None:
        """
        it is only allowed to do +-*/ with int or float or bf_array in the following order:

        bf_array + int
        bf_array + float
        bf_array + bf_array (either the same shape or (1,))

        each instance of bf_array has the following properties:
        - context - to specify precision/rounding
        - shape - shape of the array
        - array - the array itself, it is of type np.array, dtype = object, full of Bigfloat number objects

        can only create bf_array from bf_array or BigFloat number or numpy array

        """


        self.context = context

        with self.context:

            if isinstance(array, bf_array):
                # print("creating bf_array from bf_array")
                self.shape = np.shape(array.array)
                self.array = np.empty(shape=self.shape, dtype=object)
                if len(self.shape) == 1:
                    for i in range(self.shape[0]):
                        self.array[i] = array.array[i]
                elif len(self.shape) == 2:
                    for i in range(self.shape[0]):
                        for j in range(self.shape[1]):
                            self.array[i, j] = array.array[i, j]
            elif isinstance(array, bf.BigFloat):
                # print("creating bf_array from BigFloat number")
                self.array = np.empty((1,), dtype=object)
                self.array[0] = array
                self.shape = np.shape(self.array)
            else:
                # print("creating bf_array from (standard) array")
                self.shape = np.shape(array)
                self.array = np.empty(shape=self.shape, dtype=object)

                if len(self.shape) == 1:
                    if isinstance(array[0], bf.BigFloat):
                        # print("creating bf_array from BigFloat data type")
                        for i in range(self.shape[0]):
                            self.array[i] = array[i]
                    else:
                        # print("creating bf_array from standar data type")
                        for i in range(self.shape[0]):
                            self.array[i] = bf.BigFloat(array[i], context=context)

                elif len(self.shape) == 2:
                    if isinstance(array[0, 0], bf.BigFloat):
                        # print("creating bf_array from BigFloat data type")
                        for i in range(self.shape[0]):
                            for j in range(self.shape[1]):
                                self.array[i, j] = array[i, j]
                    else:
                        # print("creating bf_array from standar data type")
                        for i in range(self.shape[0]):
                            for j in range(self.shape[1]):
                                self.array[i, j] = bf.BigFloat(array[i, j], context=context)
    
    def print(self):
        if len(self.shape) == 1:
            for i in range(self.shape[0]):
                print(self.array[i], end=" ")
            print("\n")
            print("precision =", self.array[0].precision)
        elif len(self.shape) == 2:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    print(self.array[i, j], end=" ")
                print("\n")
            print("\n")
            print("precision =", self.array[0, 0].precision)

    def __add__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            # print("sum with int or float")

            shape_self = self.shape
            result = np.empty(shape=shape_self, dtype=object)
            if len(shape_self) == 1:
                for i in range(shape_self[0]):
                    result[i] = bf.add(self.array[i], bf.BigFloat(other, context=self.context), context=self.context)

            elif len(shape_self) == 2:
                for i in range(shape_self[0]):
                    for j in range(shape_self[1]):
                        result[i, j] = bf.add(self.array[i, j], bf.BigFloat(other, context=self.context), context=self.context)

            return bf_array(result)

        elif isinstance(other, bf_array):
            # print("sum with bf_array")

            shape_self = self.shape
            shape_other = other.shape

            if (shape_self == shape_other):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.add(self.array[i], other.array[i], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.add(self.array[i, j], other.array[i, j], context=self.context)

                return bf_array(result)

            elif (shape_other == (1,)):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.add(self.array[i], other.array[0], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.add(self.array[i, j], other.array[0], context=self.context)
                
                return bf_array(result)

        else:
            print("ERROR: cannot sum arrays of very different shapes")
            return -1
    
    def __sub__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            # print("sub with int or float")

            shape_self = self.shape
            result = np.empty(shape=shape_self, dtype=object)
            if len(shape_self) == 1:
                for i in range(shape_self[0]):
                    result[i] = bf.sub(self.array[i], bf.BigFloat(other, context=self.context), context=self.context)

            elif len(shape_self) == 2:
                for i in range(shape_self[0]):
                    for j in range(shape_self[1]):
                        result[i, j] = bf.sub(self.array[i, j], bf.BigFloat(other, context=self.context), context=self.context)

            return bf_array(result)

        elif isinstance(other, bf_array):
            # print("sub with bf_array")

            shape_self = self.shape
            shape_other = other.shape

            if (shape_self == shape_other):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.sub(self.array[i], other.array[i], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.sub(self.array[i, j], other.array[i, j], context=self.context)
                
                return bf_array(result)

            elif (shape_other == (1,)):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.sub(self.array[i], other.array[0], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.sub(self.array[i, j], other.array[0], context=self.context)
                
                return bf_array(result)

        else:
            print("ERROR: cannot sub arrays of very different shapes")
            return -1
    
    def __mul__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            # print("mul with int or float")

            shape_self = self.shape
            result = np.empty(shape=shape_self, dtype=object)
            if len(shape_self) == 1:
                for i in range(shape_self[0]):
                    result[i] = bf.mul(self.array[i], bf.BigFloat(other, context=self.context), context=self.context)

            elif len(shape_self) == 2:
                for i in range(shape_self[0]):
                    for j in range(shape_self[1]):
                        result[i, j] = bf.mul(self.array[i, j], bf.BigFloat(other, context=self.context), context=self.context)

            return bf_array(result)

        elif isinstance(other, bf_array):
            # print("mul with bf_array")

            shape_self = self.shape
            shape_other = other.shape

            if (shape_self == shape_other):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.mul(self.array[i], other.array[i], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.mul(self.array[i, j], other.array[i, j], context=self.context)
                
                return bf_array(result)

            elif (shape_other == (1,)):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.mul(self.array[i], other.array[0], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.mul(self.array[i, j], other.array[0], context=self.context)
                
                return bf_array(result)

        else:
            print("ERROR: cannot mul arrays of very different shapes")
            return -1
    
    def __truediv__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            # print("div with int or float")

            shape_self = self.shape
            result = np.empty(shape=shape_self, dtype=object)
            if len(shape_self) == 1:
                for i in range(shape_self[0]):
                    result[i] = bf.div(self.array[i], bf.BigFloat(other, context=self.context), context=self.context)

            elif len(shape_self) == 2:
                for i in range(shape_self[0]):
                    for j in range(shape_self[1]):
                        result[i, j] = bf.div(self.array[i, j], bf.BigFloat(other, context=self.context), context=self.context)

            return bf_array(result)

        elif isinstance(other, bf_array):
            # print("div with bf_array")

            shape_self = self.shape
            shape_other = other.shape

            if (shape_self == shape_other):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.div(self.array[i], other.array[i], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.div(self.array[i, j], other.array[i, j], context=self.context)
                
                return bf_array(result)

            elif (shape_other == (1,)):
                result = np.empty(shape=shape_self, dtype=object)
                if len(shape_self) == 1:
                    for i in range(shape_self[0]):
                        result[i] = bf.div(self.array[i], other.array[0], context=self.context)

                elif len(shape_self) == 2:
                    for i in range(shape_self[0]):
                        for j in range(shape_self[1]):
                            result[i, j] = bf.div(self.array[i, j], other.array[0], context=self.context)
                
                return bf_array(result)

        else:
            print("ERROR: cannot div arrays of very different shapes")
            return -1


def exp(array: bf_array):
    # print("exp of bf_array")

    shape_self = array.shape

    result = np.empty(shape=shape_self, dtype=object)
    if len(shape_self) == 1:
        for i in range(shape_self[0]):
            result[i] = bf.exp(array.array[i], context=array.context)

    elif len(shape_self) == 2:
        for i in range(shape_self[0]):
            for j in range(shape_self[1]):
                result[i, j] = bf.exp(array.array[i, j], context=array.context)

    return bf_array(result)

def log(array: bf_array):  # log = ln
    # print("natural log of bf_array")

    shape_self = array.shape

    result = np.empty(shape=shape_self, dtype=object)
    if len(shape_self) == 1:
        for i in range(shape_self[0]):
            result[i] = bf.log(array.array[i], context=array.context)

    elif len(shape_self) == 2:
        for i in range(shape_self[0]):
            for j in range(shape_self[1]):
                result[i, j] = bf.log(array.array[i, j], context=array.context)

    return bf_array(result)

def sum(array: bf_array):

    # print("sum of bf_array")

    shape_self = array.shape

    result = np.empty((1,), dtype=object)
    result[0] = bf.BigFloat(0, context=array.context)
    if len(shape_self) == 1:
        for i in range(shape_self[0]):
            result[0] = bf.add(result[0], array.array[i], context=array.context)

    elif len(shape_self) == 2:
        for i in range(shape_self[0]):
            for j in range(shape_self[1]):
                result[0] = bf.add(result[0], array.array[i, j], context=array.context)

    return bf_array(result)

def norm(array: bf_array):

    # print("norm of bf_array")

    shape_self = array.shape

    result = np.empty((1,), dtype=object)
    result[0] = bf.BigFloat(0, context=array.context)
    if len(shape_self) == 1:
        for i in range(shape_self[0]):
            result[0] = bf.add(result[0], bf.pow(array.array[i], 2, context=array.context), context=array.context)

    elif len(shape_self) == 2:
        for i in range(shape_self[0]):
            for j in range(shape_self[1]):
                result[0] = bf.add(result[0], bf.pow(array.array[i, j], 2, context=array.context), context=array.context)

    result[0] = bf.sqrt(result[0], context=array.context)
    return bf_array(result)




# n = np.random.normal(loc=0, scale=1, size=(2, 2))
# a = bf_array(n)
# a.print()
# (a + 2).print()
# (a - 2).print()
# (a * 2).print()
# (a / a).print()
# (a * a).print()



# n = np.ones((2, 2))
# print(n)

# a = bf_array(n)
# a.print()

# b = exp(a)
# b.print()

# log(b).print()

# c = bf_array(b.array[0])
# c.print()

# d = sum(c)
# d.print()

# (d / d).print()

# (norm(c)).print()

# num1 = bf_array(bf.BigFloat(42, context=bf.precision(200)))
# num2 = bf_array(bf.BigFloat(69, context=bf.precision(200)))
# num3 = num1 + num2
# num3.print()

# c.print()
# (c * num1).print()

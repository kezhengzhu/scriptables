

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("Can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)

class Vart(object):
    _order = 1

    @classproperty
    def order(cls):
        return cls._order

    @order.setter
    def order(cls, value):
        checkerr(isinstance(value, int) and order > 0, "Order must be set to a positive integer")
        cls._order = value

    def __init__(self, value):
        self.val = value

class TestCls(object):
    _table = [[]]
    _total = 0
    def __init__(self):
        self.index = TestCls._total+1
        TestCls._total += 1

    @classmethod
    def get_total(cls):
        return cls._total

A = TestCls()
print(A.index, TestCls.get_total())

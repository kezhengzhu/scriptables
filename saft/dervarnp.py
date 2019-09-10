

import math
import numpy as np

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

def checkwarn(cond, message):
    if not (cond):
        print("#"*32)
        print(message)
        print("#"*32)

class Var(object):
    order = 1
    count = 0
    count_var = False
    def __init__(self, value):
        checkerr(isinstance(value, (int, float, np.ndarray)), "Use float values for Var object")
        self.value = value
        self.children = []
        self.grad_val = None
        self.on = False
        if Var.count_var: Var.count += 1
        self.isreset = True

    def __repr__(self):
        if isinstance(self.value, np.ndarray):
            return self.value.__repr__() + ' with grad'
        return f'{self.value:11.3e}' + (f' with grad{self.grad_val:11.3e}' if self.grad_val is not None else '')

    def set_subject(self):
        self.isreset = False
        if isinstance(self.value, np.ndarray):
            self.grad_val = np.ones(self.value.shape)
        else:
            self.grad_val = 1.

    grad_count = 0
    count_grad = False
    def grad(self, getvar=False, order=None, depth=1):
        if self.on == True:
            raise Exception("Error: Var object is children of itself. Infinite recursion will occur if grad runs")
        self.on = True
        if order == None: order = Var.order
        if Var.count_grad: Var.grad_count += 1
        getnextvar = False if order == 1 else True
        if self.grad_val is None:
            self.grad_val = sum(vmul(coeff, var.grad(getvar=True, order=order-1), order=order) for (coeff, var) in self.children)
            self.isreset = False
        self.on = False
        if (not getvar) and isinstance(self.grad_val, Var):
            return self.grad_val.value
        return Var(self.grad_val) if getvar and (not isinstance(self.grad_val, Var)) else self.grad_val

    reset_count = 0
    count_reset = False
    def reset(self):
        if self.on: raise Exception("Error: Var object is children of itself. Infinite recursion will occur if reset runs")
        if Var.count_reset: Var.reset_count += 1
        if self.isreset: return
        self.on = True
        self.grad_val = None
        for (coeff, var) in self.children:
            var.reset()
        self.isreset = True
        self.on = False

    @classmethod
    def set_order(cls, order):
        cls.order = order
        cls.func_order(cls.order)

    @staticmethod
    def func_order(order):
        '''
        Edits the defaults of all operations on Vars
        '''
        vmul.__defaults__ = (order,)
        sin.__defaults__ = (order,)
        cos.__defaults__ = (order,)
        sinh.__defaults__ = (order,)
        cosh.__defaults__ = (order,)
        tan.__defaults__ = (order,)
        log10.__defaults__ = (order,)
        ln.__defaults__ = (order,)
        log.__defaults__ = (None, order)
        vpow.__defaults__ = (order,)
        sqrt.__defaults__ = (order,)
        exp.__defaults__ = (order,)

    def __add__(self, other):
        '''
        Var addition: applying chain rule and adding result as children of current Var
        '''
        checkerr(isinstance(other, (Var, int, float, np.ndarray)), "Var addition has to be applied onto other Vars or int/floats")

        if isinstance(other, (int, float, np.ndarray)):
            if isinstance(self.value, np.ndarray) and isinstance(other, np.ndarray):
                checkerr(self.value.shape == other.shape, "numpy array operations require array shape to be identical")
            z = Var(self.value + other)
            self.children.append((1., z))
            return z
        
        if isinstance(self.value, np.ndarray) and isinstance(other.value, np.ndarray):
            checkerr(self.value.shape == other.value.shape, "numpy array operations require array shape to be identical")
        
        z = Var(self.value + other.value)
        self.children.append((1., z))
        other.children.append((1., z))
        
        return z
    __radd__ = __add__

    def __mul__(self, other):
        return vmul(self, other)
    __rmul__ = __mul__

    def __sub__(self, other):
        '''
        Var subtraction: applying chain rule and adding result as children of current Var
        '''
        checkerr(isinstance(other, (Var, int, float, np.ndarray)), "Var subtraction has to be applied onto other Vars or int/floats")

        if isinstance(other, (int, float, np.ndarray)):
            if isinstance(self.value, np.ndarray) and isinstance(other, np.ndarray):
                checkerr(self.value.shape == other.shape, "numpy array operations require array shape to be identical")
            z = Var(self.value - other)
            self.children.append((1., z))
            return z

        if isinstance(self.value, np.ndarray) and isinstance(other.value, np.ndarray):
            checkerr(self.value.shape == other.value.shape, "numpy array operations require array shape to be identical")

        z = Var(self.value - other.value) # this should work for np.ndarray also
        self.children.append((1., z))
        other.children.append((-1., z))
        
        return z

    def __rsub__(self, other):
        '''
        Var reverse subtraction: applying chain rule and adding result as children of current Var
        Technically, for reverse implementation I will only need float, since Var - Var will call __sub__
        '''
        checkerr(isinstance(other, (Var, int, float, np.ndarray)), "Var subtraction has to be applied onto other Vars or floats")

        if isinstance(other, (int, float, np.ndarray)):
            if isinstance(self.value, np.ndarray) and isinstance(other, np.ndarray):
                checkerr(self.value.shape == other.shape, "numpy array operations require array shape to be identical")
            z = Var(other - self.value)
            self.children.append((-1., z))
            return z

        if isinstance(self.value, np.ndarray) and isinstance(other.value, np.ndarray):
            checkerr(self.value.shape == other.value.shape, "numpy array operations require array shape to be identical")
        
        z = Var(other.value - self.value)
        self.children.append((-1., z))
        other.children.append((1., z))
        
        return z

    def __truediv__(self, other):
        '''
        Var divide uses product rule and pow of -1 instead of quotient rule (same thing tho)
        '''
        checkerr(isinstance(other, (Var, int, float, np.ndarray)), "Var divide has to be applied onto other Vars or floats")

        if isinstance(other, (int, float, np.ndarray)):
            if isinstance(self.value, np.ndarray) and isinstance(other, np.ndarray):
                checkerr(self.value.shape == other.shape, "numpy array operations require array shape to be identical")

            z = Var(self.value / other)
            self.children.append((1/other, z))
            return z

        # np array checks for pow will be done in pow
        return self * pow(other,-1)

    def __rtruediv__(self, other):
        '''
        Var divide uses product rule and pow of -1 instead of quotient rule (same thing tho)
        Technically, for reverse implementation I will only need float, since Var - Var will call __div__
        '''
        checkerr(isinstance(other, (Var, int, float, np.ndarray)), "Var divide has to be applied onto other Vars or floats")

        # if isinstance(other,int) or isinstance(other, float):
        #     z = Var(other / self.value)
        #     selfv = self if Var.order > 1 else self.value
        #     nxtorder = max(Var.order-1, 1)
        #     self.children.append((-other * vpow(selfv, -2, order=nxtorder), z))
        #     return z
        
        return other * pow(self,-1)

    def __pow__(self, other):
        return vpow(self, other)

    def __rpow__(self, other):
        return vpow(other, self)

    def __neg__(self):
        z = Var(-self.value)
        self.children.append((-1, z))

        return z

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.value < other
        checkerr(isinstance(other, Var), "Var comparison works with only other Vars and int/floats")
        return self.value < other.value

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.value <= other
        checkerr(isinstance(other, Var), "Var comparison works with only other Vars and int/floats")
        return self.value <= other.value

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        checkerr(isinstance(other, Var), "Var comparison works with only other Vars and int/floats")
        return self.value == other.value

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.value > other
        checkerr(isinstance(other, Var), "Var comparison works with only other Vars and int/floats")
        return self.value > other.value

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.value >= other
        checkerr(isinstance(other, Var), "Var comparison works with only other Vars and int/floats")
        return self.value >= other.value

def vmul(x, y, order=Var.order):
    '''
    Var multiplication: applying product rule by taking the other value and implementing as coefficient
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var multiplication has to be applied onto other Vars or int/floats")
    checkerr(isinstance(y, (Var, int, float, np.ndarray)), "Var multiplication has to be applied onto other Vars or int/floats")
    # Case both not Var
    if not (isinstance(x, Var) or isinstance(y, Var)):
        return x * y
    # Case only x is Var
    if isinstance(y, (int, float, np.ndarray)):
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            checkerr(x.value.shape == y.shape, "numpy array operations require array shape to be identical")

        z = Var(x.value * y)
        x.children.append((y, z))
        return z

    # Case only y is Var
    if isinstance(x, (int, float, np.ndarray)):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            checkerr(y.value.shape == x.shape, "numpy array operations require array shape to be identical")

        z = Var(x * y.value)
        y.children.append((x, z))
        return z

    # Case both x and y is Var
    if isinstance(x.value, np.ndarray) and isinstance(y.value, np.ndarray):
        checkerr(x.value.shape == y.value.shape, "numpy array operations require array shape to be identical")

    xv = x if order > 1 else x.value
    yv = y if order > 1 else y.value
    z = Var(x.value * y.value)

    x.children.append((yv, z))
    y.children.append((xv, z))
    return z

def sin(x, order=Var.order): # Var.order default is 1
    '''
    sin for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var sin function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.sin(x)
    if isinstance(x, np.ndarray): return np.sin(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.sin(x.value)
    if isinstance(x.value, np.ndarray): zval = np.sin(x.value)

    z = Var(zval)
    x.children.append((cos(xv, order=order-1), z))
    return z

def cos(x, order=Var.order):
    '''
    cos for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var cos function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.cos(x)
    if isinstance(x, np.ndarray): return np.cos(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.cos(x.value)
    if isinstance(x.value, np.ndarray): zval = np.cos(x.value)

    z = Var(zval)
    x.children.append((-sin(xv, order=order-1), z))
    return z

def sinh(x, order=Var.order): # Var.order default is 1
    '''
    sinh for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var sin function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.sinh(x)
    if isinstance(x, np.ndarray): return np.sinh(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.sinh(x.value)
    if isinstance(x.value, np.ndarray): zval = np.sinh(x.value)

    z = Var(zval)
    x.children.append((cosh(xv, order=order-1), z))
    return z

def cosh(x, order=Var.order):
    '''
    cos for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var cos function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.cosh(x)
    if isinstance(x, np.ndarray): return np.cosh(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.cosh(x.value)
    if isinstance(x.value, np.ndarray): zval = np.cosh(x.value)

    z = Var(zval)
    x.children.append((sinh(xv, order=order-1), z))
    return z

def tan(x, order=Var.order):
    '''
    tan for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var tan function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.tan(x)
    if isinstance(x, np.ndarray): return np.tan(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.tan(x.value)
    if isinstance(x.value, np.ndarray): zval = np.tan(x.value)

    z = Var(zval)
    x.children.append((vpow(cos(xv, order=order-1),-2, order=order-1), z))
    return z

def log10(x, order=Var.order):
    '''
    log10 for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var log10 function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.log10(x)
    if isinstance(x, np.ndarray): return np.log10(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.log10(x.value)
    if isinstance(x.value, np.ndarray): zval = np.log10(x.value)

    z = Var(zval)
    x.children.append((vpow(xv*math.log(10.),-1, order=order-1), z)) # Multiplication here is with a constant, so doesn't matter
    return z

def ln(x, order=Var.order):
    '''
    ln for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var ln function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.log(x)
    if isinstance(x, np.ndarray): return np.log(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.log(x.value)
    if isinstance(x.value, np.ndarray): zval = np.log(x.value)

    z = Var(zval)
    x.children.append((vpow(xv,-1, order=order-1), z))
    return z

def log(x, base=None, order=Var.order):
    '''
    log for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var log function has to be applied onto Vars/ints/floats only")
    checkerr(base is None or isinstance(base, (int, float)), "Var log function base has to be float or int (implementation of Var is not available yet)")

    if isinstance(x, (int, float)): return math.log(x) if base is None else math.log(x,base)
    if isinstance(x, np.ndarray): return np.log(x) if base is None else (np.log(x) / math.log(base))

    # Natural log or ln
    if base is None:
        return ln(x, order=order)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.log(x.value, base)
    if isinstance(x.value, np.ndarray): zval = np.log(x.value) / math.log(base)

    z = Var(zval)
    x.children.append((vpow(xv*math.log(base), -1, order=order-1), z))
    return z

def sqrt(x, order=Var.order):
    '''
    sqrt for Var objects only
    '''
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var sqrt function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.sqrt(x)
    if isinstance(x, np.ndarray): return np.sqrt(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.sqrt(x.value)
    if isinstance(x.value, np.ndarray): zval = np.sqrt(x.value)

    z = Var(zval)
    x.children.append((0.5*vpow(xv, -0.5, order=order-1), z))
    return z

def vpow(x, y, order=Var.order):
    ''' 
    pow for Var and int/float for polynomials, and also for Var and Var for power functions
    '''
    checkerr(isinstance(y, (Var, int, float, np.ndarray)),
        "Object of pow function has to be Var object or int or float")
    checkerr(isinstance(x, (Var, int, float, np.ndarray)),
        "Subject of pow function has to be Var object or int or float")
    if order > 1:
        xv = x
        yv = y
    else:
        xv = x.value if isinstance(x, Var) else x
        yv = y.value if isinstance(y, Var) else y
    # Case if both is int/float
    if not (isinstance(y, Var) or isinstance(x, Var)):
        return pow(x,y) # this should work for np.ndarray too
    # Case x is Var, y is int/float
    if isinstance(y, (int, float, np.ndarray)):
        if (not isinstance(y, np.ndarray)) and (y == 0 or y == 0.):
            # what happens if np.ndarray has to power of 0?
            return Var(1.)
        if isinstance(x.value, np.ndarray) and isinstance(y, np.ndarray):
            checkerr(x.value.shape == y.shape, "numpy array operations require array shape to be identical")
        z = Var(pow(x.value, y))
        x.children.append((vpow(xv, y-1, order=order-1)*y,z)) # Var * np.ndarray calls vmul, but np.ndarray * Var calls np implementation <-- avoid. 
                                                              # Because it doesn't recognise so it tries to iterate instead
        return z
    # Case y is Var, x is int/float
    if isinstance(x, (int, float, np.ndarray)):
        if isinstance(x, np.ndarray) and isinstance(y.value, np.ndarray):
            checkerr(x.shape == y.value.shape, "numpy array operations require array shape to be identical")
        z = Var(pow(x, y.value))
        xlog = np.log(x) if isinstance(x, np.ndarray) else math.log(x)
        y.children.append((vpow(x, yv, order=order-1)*xlog, z))
        return z
    # Case both is Var
    if y.value == 0 or y.value == 0.:
        return Var(1.)
    if isinstance(x.value, np.ndarray) and isinstance(y.value, np.ndarray):
        checkerr(x.value.shape == y.value.shape, "numpy array operations require array shape to be identical")
    z = Var(pow(x.value, y.value))
    x.children.append((vmul(yv,vpow(xv, yv-1, order=order-1), order=order-1), z)) # Here multiplication has to use vmul to pull in the order
    y.children.append((vmul(ln(xv, order=order-1),vpow(xv, yv, order=order-1), order=order-1), z))

    return z

def exp(x, order=Var.order):
    checkerr(isinstance(x, (Var, int, float, np.ndarray)), "Var exp function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x, (int, float)): return math.exp(x)
    if isinstance(x, np.ndarray): return np.exp(x)

    xv = x if order > 1 else x.value

    if isinstance(x.value, (int, float)): zval = math.exp(x.value)
    if isinstance(x.value, np.ndarray): zval = np.exp(x.value)

    z = Var(zval)
    x.children.append((exp(xv, order=order-1), z))

    return z

def derivative(y, *args, order=1, getvar=False):
    '''
    Calculates dy/dx via automatic differentiation, for any amount of x
    '''
    checkerr(order <= (Var.order-getvar) and isinstance(getvar, bool), "Order must less than or equal to the set order. If getvar=True, it must be strictly less than.")

    result = tuple()

    thisvar = True
    exes = args
    get_op = lambda x, b1: x if b1 or (not isinstance(x,Var)) else x.value

    subj = [y]
    reduc = list(range(len(args)))
    for i in range(order):
        der = i+1
        nxtsubj = []
        nxtreduc = []
        if der == order and (not getvar): thisvar = False
        for fx in range(len(subj)):
            for x in args:
                x.reset()
            subj[fx].set_subject()
            nxt = [x.grad(getvar=thisvar, order=order-i) for x in args[reduc[fx]:len(args)]]
            nxtsubj += nxt
            nxtreduc += list(range(reduc[fx],len(args)))
        subj = nxtsubj
        reduc = nxtreduc
        if len(nxtsubj) == 1: result = result + (get_op(nxtsubj[0], getvar),)
        else: result = result + ([get_op(j, getvar) for j in nxtsubj],)
    for x in args:
        x.reset()

    if len(result) == 1: 
        return result[0]
    return result

def jacobian(func, x, args=(), kwargs={}, getvar=False):
    '''
    Returns jacobian of a given function with AD
    Function parameters: return either a value or a list of value, taking in args
    such that: func(x, args, kwargs)
    '''
    Var.set_order(1)
    # First run to determine output size
    out = func(x, *args, **kwargs)
    if isinstance(out, (int, float)):
        flen = 0
    elif isinstance(out, list) and all(isinstance(y, (int, float)) for y in out):
        flen = len(out)
    else:
        raise Exception("Jacobian output function should return integer/float or list of values")

    # Next generate the output Vars
    if isinstance(x, list) and all(isinstance(xi, (int, float)) for xi in x):
        xl_ = []
        for xval in x:
            xl_.append(Var(xval))
        yl_ = func(xl_, *args, **kwargs)
    elif isinstance(x, (int, float)):
        xl_ = [Var(x)]
        yl_ = func(xl_[0], *args, **kwargs)
    else:
        raise Exception("Jacobian inputs must be integer/float or list of values")

    if flen == 0:
        return derivative(yl_, *xl_, order=1, getvar=False)

    result = []
    for i in range(flen):
        yout = derivative(yl_[i], *xl_, order=1, getvar=False)
        result.append(list(yout))

    return result

############# Below this is testing ##############

def testfn(x, y, var, dorder=1):
    x.reset()
    y.reset()
    var.grad_val = 1.
    proceed = True if dorder > 1 else False
    dzdx = x.grad(getvar=proceed, order=dorder)
    dzdy = y.grad(getvar=proceed, order=dorder)
    result = []

    if proceed:
        result.append(dzdx.value)
        dxnext = testfn(x,y,dzdx,dorder=dorder-1)
        result.append(dxnext)
    else:
        result.append(dzdx)

    if proceed:
        result.append(dzdy.value)
        dynext = testfn(x,y,dzdy,dorder=dorder-1)
        result.append(dynext)
    else:
        result.append(dzdy)

    return result

def plist(L1, L2, prefix=""):
    for i in range(len(L1)):
        if isinstance(L1[i], list):
            xy = 'x' if i==1 else 'y'
            plist(L1[i], L2[i], prefix=prefix+xy+'--')
        else:
            xy = 'x' if i==0 else 'y'
            print(prefix+xy,f'{L1[i]:6.3f}, {L2[i]:6.3f}')

def mulfunc(x):
    assert(len(x) == 3)
    y = [0] * 4
    y[0] = x[0] ** 2 - 2 * x[1]
    y[1] = -x[0] ** 3 + 3 * x[2] ** 2 - x[1]
    y[2] = x[1] ** 2 - 4 * x[0] + exp(-x[2]) / (1 - x[2])
    y[3] = exp(-(x[0]+2*x[1]-x[2])**2) /(1 - (x[0] + x[1] + x[2]))

    return y

def mulfunc_d(x):
    assert(len(x) == 3)
    dy = [0] * 4
    dy[0] = [2 * x[0], -2., 0.]
    dy[1] = [-3 * x[0]**2, -1., 6 * x[2]]
    dy[2] = [-4., 2*x[1], x[2] * exp(-x[2]) / (1-x[2])**2]
    y4 = exp(-(x[0]+2*x[1]-x[2])**2) /(1 - (x[0] + x[1] + x[2]))
    dy[3] = [-2*(x[0]+2*x[1]-x[2]) * y4 + y4 / (1 - (x[0] + x[1] + x[2])),
             -4*(x[0]+2*x[1]-x[2]) * y4 + y4 / (1 - (x[0] + x[1] + x[2])),
              2*(x[0]+2*x[1]-x[2]) * y4 + y4 / (1 - (x[0] + x[1] + x[2]))]

    return dy

def main():
    print('{:18s}'.format("Testing numpy var..."))
    print('='*18)
    Var.set_order(2)
    a = np.array([0.21, 0.63, 1.56])
    a_ = Var(a)
    print(a_ * 2.)
    a1 = a_ * np.ones(a.shape)
    print(a1, a_.children)
    b = np.array([1., 2., 0.5])
    print(a_ * b)
    print(a_ / b)
    b_ = Var(b)
    pwr = np.array([1,2,3])
    test = 6.1 * b_** pwr - 5 * sin(0.325 * a_**3) - exp(-a_**2 * b_**0.5)
    test.set_subject()
    der = derivative(test,a_, b_,order=2)
    print('{:18s}'.format("First a_ derivatives: "), der[0][0])
    print('{:18s}'.format("First a_ der actual: "), - 5 * 0.325 * 3 * a**2 * cos(0.325*a**3) + 2 * a * b**0.5 * exp(-a**2 * b**0.5))
    print('{:18s}'.format("Second a_ derivatives: "), der[1][0])
    print('{:18s}'.format("Second a_ der actual: "), 5 * (0.325 * 3 * a**2)**2 * sin(0.325*a**3) - 5 * 0.325 * 3 * 2 * a * cos(0.325*a**3) - (2 * a * b**0.5)**2 * exp(-a**2 * b**0.5) + 2 * b**0.5 * exp(-a**2 * b**0.5))
    print()
    print('{:18s}'.format("First b_ derivatives: "), der[0][1])
    print('{:18s}'.format("First b_ der actual: "), 6.1 * pwr * b**(pwr-1) + 0.5*b**(-0.5) * a**2 * exp(-a**2 * b**0.5))
    print('{:18s}'.format("Second b_ derivatives: "), der[1][2])
    print('{:18s}'.format("Second b_ der actual: "), 6.1 * pwr * (pwr-1) * b**(pwr-2) - (0.5*b**(-0.5) * a**2)**2 * exp(-a**2 * b**0.5) - 0.5 * 0.5*b**(-1.5) * a**2 * exp(-a**2 * b**0.5))
    print()
    print('{:18s}'.format("Second dab_ derivatives: "), der[1][1])
    print('{:18s}'.format("Second dab_ der actual: "), - a**3 * exp(-a**2 * b**0.5) + a * b**(-0.5) * exp(-a**2 * b**0.5))
    print('='*18)

    # print('sin test')
    # Var.set_order(3)
    # x = Var(0.71)
    # y = Var(2)
    # z = 1/x
    # print(derivative(z,x,order=3))
    # x = 0.71
    # print((-1/x**2, 2/x**3, -6/x**4))

    # x=Var(0.71)
    # z2 = x**3.1
    # print(derivative(z2, x, order=3))
    # x = 0.71
    # print(3.1*x**2.1, 3.1*2.1*x**1.1, 3.1*2.1*1.1*x**0.1)

    # x=Var(0.71)
    # z2 = sin(x)
    # z2.set_subject()
    # dx = x.grad(getvar=True, order=3)
    # x.reset()
    # print(z2.grad_val)
    # dx.set_subject()
    # dx2 = x.grad(getvar=True, order=2)
    # x.reset()
    # dx2.set_subject()
    # dx3 = x.grad(getvar=False, order=1)
    # print(dx.value, dx2.value, dx3)

    # x = Var(0.71)
    # z2 = sin(x)
    # print(x.children)
    # print(derivative(z2, x, order=3))

    # x = 0.71
    # print(cos(x), -sin(x), -cos(x))
    # fx = lambda x: exp(-sin(x) / x**2) - 1
    # print(jacobian(fx, 12), -(cos(12)/12**2 - 2*sin(12)/12**3)*exp(-sin(12)/12**2))
    # xL = jacobian(mulfunc, [0.31, 0.12, 0.74])
    # dyL = mulfunc_d([0.31, 0.12, 0.74])
    # for L in xL: 
    #     print(L)
    # for L in dyL:
    #     print(L)
    # zs = np.zeros(100)
    # dz = np.zeros(100)
    # d2z = np.zeros(100)
    # xs = np.linspace(0.1,1,100)
    # for i in range(len(xs)):
    #     x = Var(xs[i])
    #     z = y * exp(-x * sin(x*0.31)) + pow(x,1.3223)*exp(-y*cos(x)) - x**0.34/y**2.1 + x**0.92/y**1.42
    #     dz[i], d2z[i], _ = derivative(z,x, order=3)
    #     zs[i] = z.value


    # f = lambda x, y : y * np.exp(-x * np.sin(x*0.31)) + pow(x,1.3223)*np.exp(-y*np.cos(x)) - x**0.34/y**2.1 + x**0.92/y**1.42
    
    # df = lambda x, y : (f(x*(1+1e-5), y) - f(x*(1-1e-5), y)) / (2e-5*x)
    # d2f = lambda x, y: (df(x*(1+1e-5 ),y) - df(x*(1-1e-5),y)) / (2e-5*df(x,y))

    # fig, ax = plt.subplots()
    # xp = np.linspace(0.1,1,100)
    # zp = f(xp,2)
    # dzp = df(xp,2)
    # d2zp = d2f(xp,2)
    # ax.plot(xp, dzp, 'r.')
    # ax.plot(xp, d2z, 'b-')
    # plt.show()
    # dz = testfn(x,y,z,dorder=3)
    # x,y = (0.71, 2)
    # dx1,dy1 = (2*y**2*cos(2*x), 2*y*sin(2*x))
    # dx2,dxy,dy2 = (-4*y**2*sin(2*x), 4*y*cos(2*x), 2*sin(2*x))
    # dx3,dx2y,dxy2,dy3 = (-8*y**2*cos(2*x), -8*y*sin(2*x), 4*cos(2*x), 0.)
    # dx4,dx3y,dx2y2,dxy3,dy4 = (16*y**2*sin(2*x), -16*y*cos(2*x), -8*sin(2*x), 0., 0.) 
    # L = [dx1,[dx2,[dx3,[dx4,dx3y],dx2y,[dx3y,dx2y2]],dxy,[dx2y,[dx3y,dx2y2],dxy2,[dx2y2,dxy3]]],dy1,[dxy,[dx2y,[dx3y,dx2y2],dxy2,[dx2y2,dxy3]],dy2,[dxy2,[dx2y2,dxy3],dy3,[dxy3,dy4]]]]
    # L3 = [dx1,[dx2,[dx3,dx2y],dxy,[dx2y,dxy2]],dy1,[dxy,[dx2y,dxy2,],dy2,[dxy2,dy3]]]
    # plist(dz,L3)
    # dz1 = {}
    # dz2 = {}
    # dz3 = {}
    # for i in [1, 2, 3]:
    #     Var.set_order(i)
    #     print(f'Setting Var order to {i:d}...', '='*8)
    #     print("Test case 1: z = x^y + y^2 sin(x^3)")
    #     x = Var(0.71)
    #     y = Var(1.213)
    #     z1 = x**y + y**2 * sin(x**3)
    #     dz1[i] = testfn(x,y,z1,dorder=i)
    #     print("Test case 2: z = y^5 * exp(x^0.5) - 5 * cos x * sin x")
    #     x = Var(0.71)
    #     y = Var(1.213)
    #     z2 = y**5 * exp(sqrt(x)) - 5*cos(x)*sin(x)
    #     dz2[i] = testfn(x,y,z2,dorder=i)
    #     print("Test case 3: z = x^3 * y^2 + ln(y+x+x^2) + log10(x*y*sin x)")
    #     x = Var(0.71)
    #     y = Var(1.213) # re-defining new vars to reduce branches
    #     z3 = x**3 * y**2 + ln(y+x+x**2) + log10(x*y*sin(x))
    #     dz3[i] = testfn(x,y,z3,dorder=i)

    # for i in [1,2,3]:
    #     print(dz1[i])
    #     print(dz2[i])
    #     print(dz3[i])
    # x = 0.71
    # y = 1.213
    # # change for simplicity
    # print('Displaying results for 3 test cases against actual answer...')
    # print('='*32)
    # print("Test case 1: z = x^y + y^2 sin(x^3)")
    # print('='*32)
    # st = ['First', 'Second', 'Third']
    # for i in [1,2,3]:
    #     print(f'First derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{y*x**(y-1)+y**2*cos(x**3)*3*x**2:10.3f}{ln(x)*x**y + 2*y*sin(x**3):10.3f}')
    #     id2 = 1 if i == 1 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz1[i][0]:10.3f}{dz1[i][id2]:10.3f}')
    # print()
    # for i in [2,3]:
    #     print(f'Second derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{y * (y-1) * x**(y-2) + 6 * x * y**2 *cos(x**3) - 9 * x**4 * y**2 * sin(x**3):10.3f}{ln(x)*ln(x)*x**y+2*sin(x**3):10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz1[i][1][0]:10.3f}{dz1[i][3][id2]:10.3f}')
    #     print(f'd/dxdy derivatives: via order {i:d}')
    #     dxdy = y * ln(x) * x**(y-1) + 6 * x**2 * y * cos(x**3) + x**(y-1)
    #     print('{:20s}'.format('Actual values:'), f'{dxdy:10.3f}{dxdy:10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz1[i][1][id2]:10.3f}{dz1[i][3][0]:10.3f}')
    # print()
    # print(f'Third derivatives: via order 3')
    # x3 = x**(-3 + y)*(-2 + y)*(-1 + y)*y + 6*y**2*cos(x**3) - 27*x**6*y**2*cos(x**3) - 54*x**3*y**2*sin(x**3)
    # x2y = x**(-2 + y)*(-1 + y) + x**(-2 + y)*y + 12*x*y*cos(x**3) + x**(-2 + y)*(-1 + y)*y*log(x) - 18*x**4*y*sin(x**3)
    # xy2 = 6*x**2*cos(x**3) + 2*x**(-1 + y)*ln(x) + x**(-1 + y)*y*ln(x)**2
    # y3 = x**y*ln(x)**3
    # print('{:20s}'.format('Actual values:'), f'{x3:10.3f}{x2y:10.3f}{x2y:10.3f}{xy2:10.3f}{x2y:10.3f}{xy2:10.3f}{xy2:10.3f}{y3:10.3f}')
    # print('{:20s}'.format('Third order result:'), f'{dz1[3][1][1][0]:10.3f}{dz1[3][1][1][1]:10.3f}{dz1[3][1][3][0]:10.3f}{dz1[3][1][3][1]:10.3f}{dz1[3][3][1][0]:10.3f}{dz1[3][3][1][1]:10.3f}{dz1[3][3][3][0]:10.3f}{dz1[3][3][3][1]:10.3f}')

    # print('='*32)
    # print("Test case 2: z = y^5 * exp(x^0.5) - 5 * cos x * sin x")
    # print('='*32)
    # for i in [1,2,3]:
    #     print(f'First derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{0.5*y**5*x**(-0.5)*exp(x**0.5)-5*(cos(x)**2)+5*(sin(x)**2):10.3f}{5 * y**4 * exp(x**0.5):10.3f}')
    #     id2 = 1 if i == 1 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz2[i][0]:10.3f}{dz2[i][id2]:10.3f}')
    # print()
    # for i in [2,3]:
    #     print(f'Second derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{(-0.25*exp(x**0.5)*y**5)/x**1.5 + (0.25*exp(x**0.5)*y**5)/x**1. + 20*cos(x)*sin(x):10.3f}{20 * y**3 * exp(x**0.5):10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz2[i][1][0]:10.3f}{dz2[i][3][id2]:10.3f}')
    #     print(f'd/dxdy derivatives: via order {i:d}')
    #     dxdy = (2.5*exp(x**0.5)*y**4)/x**0.5
    #     print('{:20s}'.format('Actual values:'), f'{dxdy:10.3f}{dxdy:10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz2[i][1][id2]:10.3f}{dz2[i][3][0]:10.3f}')
    # print()
    # print(f'Third derivatives: via order 3')
    # x3 = x**(-3 + y)*(-2 + y)*(-1 + y)*y + 6*y**2*cos(x**3) - 27*x**6*y**2*cos(x**3) - 54*x**3*y**2*sin(x**3)
    # x2y = x**(-2 + y)*(-1 + y) + x**(-2 + y)*y + 12*x*y*cos(x**3) + x**(-2 + y)*(-1 + y)*y*log(x) - 18*x**4*y*sin(x**3)
    # xy2 = 6*x**2*cos(x**3) + 2*x**(-1 + y)*ln(x) + x**(-1 + y)*y*ln(x)**2
    # y3 = x**y*ln(x)**3
    # print('{:20s}'.format('Actual values:'), f'{x3:10.3f}{x2y:10.3f}{x2y:10.3f}{xy2:10.3f}{x2y:10.3f}{xy2:10.3f}{xy2:10.3f}{y3:10.3f}')
    # print('{:20s}'.format('Third order result:'), f'{dz1[3][1][1][0]:10.3f}{dz1[3][1][1][1]:10.3f}{dz1[3][1][3][0]:10.3f}{dz1[3][1][3][1]:10.3f}{dz1[3][3][1][0]:10.3f}{dz1[3][3][1][1]:10.3f}{dz1[3][3][3][0]:10.3f}{dz1[3][3][3][1]:10.3f}')

    # print('='*32)
    # print("Test case 3: z = x^3 * y^2 + ln(y+x+x^2) + log10(x*y*sin x)")
    # print('='*32)

    # for i in [1,2,3]:
    #     print(f'First derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{4.33918:10.3f}{1.63834:10.3f}')
    #     id2 = 1 if i == 1 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz3[i][0]:10.3f}{dz3[i][id2]:10.3f}')
    # print()
    # for i in [2,3]:
    #     print(f'Second derivatives: via order {i:d}')
    #     print('{:20s}'.format('Actual values:'), f'{4.21424:10.3f}{0.250903:10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz3[i][1][0]:10.3f}{dz3[i][3][id2]:10.3f}')
    #     print(f'd/dxdy derivatives: via order {i:d}')
    #     dxdy =3.25803
    #     print('{:20s}'.format('Actual values:'), f'{dxdy:10.3f}{dxdy:10.3f}')
    #     id2 = 1 if i == 2 else 2
    #     print('{:20s}'.format(f'{st[i-1]:s} order result:'), f'{dz3[i][1][id2]:10.3f}{dz3[i][3][0]:10.3f}')
    # print()
    # print(f'Third derivatives: via order 3')
    # x3 = 13.1511
    # x2y = 10.8145
    # xy2 = 3.36312
    # y3 = 0.626551
    # print('{:20s}'.format('Actual values:'), f'{x3:10.3f}{x2y:10.3f}{x2y:10.3f}{xy2:10.3f}{x2y:10.3f}{xy2:10.3f}{xy2:10.3f}{y3:10.3f}')
    # print('{:20s}'.format('Third order result:'), f'{dz3[3][1][1][0]:10.3f}{dz3[3][1][1][1]:10.3f}{dz3[3][1][3][0]:10.3f}{dz3[3][1][3][1]:10.3f}{dz3[3][3][1][0]:10.3f}{dz3[3][3][1][1]:10.3f}{dz3[3][3][3][0]:10.3f}{dz3[3][3][3][1]:10.3f}')

    # print(Var.grad_count, Var.reset_count)

if __name__ == '__main__':
    main()


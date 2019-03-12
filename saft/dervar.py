
import math

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

def checkwarn(cond, message):
    if not (cond):
        print("#"*32)
        print(message)
        print("#"*32)

class Var(object):
    def __init__(self, value):
        checkerr(isinstance(value, float), "Use float values for Var object")
        self.value = value
        self.children = []
        self.grad_val = None

    def __repr__(self):
        return f'{self.value:11.3e}' + (f' with grad{self.grad_val:11.3e}' if self.grad_val is not None else '')

    def grad(self):
        if self.grad_val is None:
            self.grad_val = sum(coeff * var.grad() for (coeff, var) in self.children)
        return self.grad_val

    def __add__(self, other):
        '''
        Var addition: applying chain rule and adding result as children of current Var
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var addition has to be applied onto other Vars or int/floats")

        if isinstance(other,int) or isinstance(other, float):
            z = Var(self.value + other)
            self.children.append((1., z))
            return z

        z = Var(self.value + other.value)
        self.children.append((1., z))
        other.children.append((1., z))
        
        return z
    __radd__ = __add__

    def __mul__(self, other):
        '''
        Var multiplication: applying product rule by taking the other value and implementing as coefficient
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var multiplication has to be applied onto other Vars or int/floats")
        if isinstance(other,int) or isinstance(other, float):
            z = Var(self.value * other)
            self.children.append((other, z))
            return z

        z = Var(self.value * other.value)
        self.children.append((other.value, z))
        other.children.append((self.value, z))

        return z
    __rmul__ = __mul__

    def __sub__(self, other):
        '''
        Var subtraction: applying chain rule and adding result as children of current Var
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var subtraction has to be applied onto other Vars or int/floats")

        if isinstance(other,int) or isinstance(other, float):
            z = Var(self.value - other)
            self.children.append((1., z))
            return z

        z = Var(self.value - other.value)
        self.children.append((1., z))
        other.children.append((-1., z))
        
        return z

    def __rsub__(self, other):
        '''
        Var reverse subtraction: applying chain rule and adding result as children of current Var
        Technically, for reverse implementation I will only need float, since Var - Var will call __sub__
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var subtraction has to be applied onto other Vars or floats")

        if isinstance(other,int) or isinstance(other, float):
            z = Var(other - self.value)
            self.children.append((-1., z))
            return z

        z = Var(other.value - self.value)
        self.children.append((-1., z))
        other.children.append((1., z))
        
        return z

    def __truediv__(self, other):
        '''
        Var divide uses product rule and pow of -1 instead of quotient rule (same thing tho)
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var divide has to be applied onto other Vars or floats")

        if isinstance(other,int) or isinstance(other, float):
            z = Var(self.value / other)
            self.children.append((1/other, z))
            return z

        return self * pow(other,-1)

    def __rtruediv__(self, other):
        '''
        Var divide uses product rule and pow of -1 instead of quotient rule (same thing tho)
        Technically, for reverse implementation I will only need float, since Var - Var will call __div__
        '''
        checkerr(isinstance(other, Var) or isinstance(other,int) or isinstance(other, float), "Var divide has to be applied onto other Vars or floats")

        if isinstance(other,int) or isinstance(other, float):
            z = Var(other / self.value)
            self.children.append((-other * pow(self.value, -2), z))
            return z
        
        return other * pow(self,-1)
    def __pow__(self, other):
        return vpow(self, other)

    def __rpow__(self, other):
        return vpow(other, self)

    def __neg__(self):
        z = Var(-self.value)
        self.children.append((-1, z))

        return z

def sin(x):
    '''
    sin for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var sin function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.sin(x)

    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z

def cos(x):
    '''
    cos for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var cos function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.cos(x)

    z = Var(math.cos(x.value))
    x.children.append((-math.sin(x.value), z))
    return z

def tan(x):
    '''
    tan for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var tan function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.tan(x)

    z = Var(math.tan(x.value))
    x.children.append((pow(math.cos(x.value),-2), z))
    return z

def log10(x):
    '''
    log10 for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var log10 function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.log10(x)

    z = Var(math.log10(x.value))
    x.children.append((1/(x.value*math.log(10.)), z))
    return z

def ln(x):
    '''
    ln for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var ln function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.log(x)

    z = Var(math.log(x.value))
    x.children.append((1/x.value, z))
    return z

def log(x, base=None):
    '''
    log for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var log function has to be applied onto Vars/ints/floats only")
    checkerr(base is None or isinstance(base, int) or isinstance(base, float), "Var log function base has to be float or int (implementation of Var is not available yet)")

    if isinstance(x,int) or isinstance(x, float): return math.log(x) if base is None else math.log(x,base)

    # Natural log or ln
    if base is None:
        return ln(x)

    z = Var(math.log(x.value, base))
    x.children.append((1/(x.value*math.log(base)), z))
    return z

def sqrt(x):
    '''
    sqrt for Var objects only
    '''
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var sqrt function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.sqrt(x)

    z = Var(math.sqrt(x.value))
    x.children.append((1/(2*math.sqrt(x.value)), z))
    return z

def vpow(x, y):
    ''' 
    pow for Var and int/float for polynomials, and also for Var and Var for power functions
    '''
    checkerr(isinstance(x,Var) or isinstance(y,Var), "Either subject or object of pow function has to be Var object")
    checkerr(isinstance(y,Var) or isinstance(y,int) or isinstance(y,float),
        "Object of pow function has to be Var object or int or float")
    checkerr(isinstance(x,Var) or isinstance(x,int) or isinstance(x,float),
        "Subject of pow function has to be Var object or int or float")
    # Case x is Var, y is int/float
    if isinstance(y,int) or isinstance(y, float):
        z = Var(math.pow(x.value, y))
        x.children.append((y*math.pow(x.value, y-1),z))
        return z
    # Case y is Var, x is int/float
    if isinstance(x,int) or isinstance(x, float):
        z = Var(math.pow(x, y.value))
        y.children.append((math.log(x)*z.value, z))
        return z

    # Case both is Var
    z = Var(math.pow(x.value, y.value))
    x.children.append((y.value*math.pow(x.value, y.value-1), z))
    y.children.append((math.log(x.value)*z.value, z))

    return z

def exp(x):
    checkerr(isinstance(x, Var) or isinstance(x,int) or isinstance(x, float), "Var exp function has to be applied onto Vars/ints/floats only")
    
    if isinstance(x,int) or isinstance(x, float): return math.exp(x)

    z = Var(math.exp(x.value))
    x.children.append((math.exp(x.value), z))

    return z

def derivative(y, x):
    '''
    Calculates dy/dx via automatic differentiation
    '''
    y.grad_val = 1.
    return x.grad()

def main():
    x = Var(0.71)
    y = Var(1.213)
    z = pow(x,y) + pow(.3,y) - pow(x,7) + (-y)
    z.grad_val = 1.
    print(f'z value is {z.value:10.4f}')
    print(f'dz/dx is {x.grad():10.4f}')
    print(f'dz/dx actual is {0.9*1/(2*x.value)-1/y.value:10.4f}')
    print(f'dz/dy is {y.grad():10.4f}')
    print(f'dz/dy actual is {x.value/y.value**2:10.4f}')


if __name__ == '__main__':
    main()



import math

class fvar(object): 
    def __init__(self, value):
        self.value = value
        self.grad = 0.

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            z = fvar(self.value + other)
            z.grad = self.grad

            return z

        z = fvar(self.value + other.value)
        z.grad = self.grad + other.grad

        return z

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            z = fvar(self.value - other)
            z.grad = self.grad

            return z

        z = fvar(self.value - other.value)
        z.grad = self.grad - other.grad

        return z

    def __rsub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            z = fvar(other - self.value)
            z.grad = -self.grad

            return z

        z = fvar(other.value - self.value)
        z.grad = other.grad - self.grad

        return z

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            z = fvar(self.value * other)
            z.grad = self.grad * other

            return z
        z = fvar(self.value * other.value)
        z.grad = other.value * self.grad + self.value * other.grad

        return z

    __rmul__ = __mul__

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            z = fvar(pow(self.value, other))
            z.grad = other * pow(self.value,(other-1)) * self.grad
            return z

        z = fvar(pow(self.value, other.value))
        z.grad = other.value * pow(self.value, other.value - 1) * self.grad + math.log(self.value) * z.value * other.grad

        return z

    def __neg__(self):
        z = fvar(-self.value)
        z.grad = -self.grad

        return z

    def __abs__(self):
        if self.value < 0:
            return -self
        return self


def sin(x):
    if isinstance(x, int) or isinstance(x, float):
        return math.sin(x)

    z = fvar(math.sin(x.value))
    z.grad = math.cos(x.value) * x.grad

    return z

def exp(x):
    if isinstance(x, (int, float)):
        return math.exp(x)

    z = fvar(math.exp(x.value))
    z.grad = math.exp(x.value) * x.grad
    return z

def main():
    # x = fvar(0.12)
    # y = fvar(1.53)

    # x.grad = 1.
    # f = lambda x, y : x*y + y * sin(x)
    # z = f(x,y)
    # print(f'z={z.value:6.4f}, dz/dx={z.grad:6.4f}, real={y.value + y.value*math.cos(x.value):6.4f}')
    # x.grad = 0.
    # y.grad = 1.
    # z = f(x,y)
    # print(f'z={z.value:6.4f}, dz/dy={z.grad:6.4f}, real={x.value + math.sin(x.value):6.4f}')

    # x.grad = 1.
    # y.grad = 0.
    # f = lambda x, y : x**y + y * sin(x)
    # z = f(x,y)
    # print(f'z={z.value:6.4f}, dz/dx={z.grad:6.4f}, real={y.value*x.value**(y.value-1) + y.value*math.cos(x.value):6.4f}')
    # x.grad = 0.
    # y.grad = 1.
    # z = f(x,y)
    # print(f'z={z.value:6.4f}, dz/dy={z.grad:6.4f}, real={math.log(x.value)*x.value**y.value+ math.sin(x.value):6.4f}')
    v = fvar(.003)
    T = fvar(300)

    R = 8.314
    a = 3e-3
    b = 3e-5

    v.grad = 1.
    T.grad = 0.

    g1 = v - b
    print(f"{'g1:':10s}{g1.value:<15.5f}{g1.grad:<15.5f}")
    g2 = g1**-1
    print(f"{'g2:':10s}{g2.value:<15.5f}{g2.grad:<15.5f}")
    g3 = R*T
    print(f"{'g3:':10s}{g3.value:<15.5f}{g3.grad:<15.5f}")
    g4 = g2 * g3
    print(f"{'g4:':10s}{g4.value:<15.5f}{g4.grad:<15.5f}")
    h1 = v**-2
    print(f"{'h1:':10s}{h1.value:<15.5f}{h1.grad:<15.5f}")
    h2 = -a * h1
    print(f"{'h2:':10s}{h2.value:<15.5f}{h2.grad:<15.5f}")
    p = g4 + h2
    print(f"{'p:':10s}{p.value:<15.5f}{p.grad:<15.5f}")

    pc = R*T * (v-b)**-1 + (-a)*v**-2
    print(pc.value, pc.grad)

if __name__ == '__main__':
    main()
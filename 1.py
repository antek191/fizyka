import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.utilities import lambdify
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


#%%
x = sp.Symbol('x', real=True)
k = sp.Symbol('k', real=True, positive=True)
a = sp.Symbol('a', real=True)
m = sp.Symbol('m', real=True, positive=True)
t = sp.Symbol('t', real=True, positive=True)
h = sp.Symbol('hbar', real=True, positive=True)
n = sp.Symbol('n', integer=True, nonzero=True)
psi = sp.Function('psi')

x, k, a, m, t, h, n, psi(x)
eq = sp.Eq(psi(x).diff(x, x), -k**2 * psi(x))
sol = sp.dsolve(eq, psi(x), ics={psi(0): 0})
sol = sol.subs(k, n * sp.pi / a)
sol = sol.subs(sp.Symbol('C1'), sp.sqrt(2 / a))
#Samo Rozwiązanie
# przesyłam samo rozwiązanie w pliku python, ponieważ nie potrefię przenieść skryptu do gita tak, aby dało się je odpalić
# plik ipynb zamieściłem w zadaniu
#Antosz Soroczyński



# Obliczanie średniej wartości położenia <x>
mean_x = sp.integrate(sol.rhs * x * sol.rhs, (x, 0, a)).simplify()

# Obliczanie średniej wartości pędu <p>
mean_p = sp.integrate(sol.rhs * sp.I * h / sp.sqrt(2 * m) * sol.rhs.conjugate(), (x, 0, a)).simplify()

# Sprawdzenie nierówności
uncertainty_relation = mean_x * mean_p >= h / 2

print("Średnia wartość położenia <x>: ", mean_x)
print("Średnia wartość pędu <p>: ", mean_p)
print("Zgodność z nierównością: ", uncertainty_relation)
import math
import time
from math import factorial
import random as rand

#ex 1
m = 1
u = 100
u2 = 1000

while 1 + u != 1:
    # print(1 + u)
    u = u / 10 #pow(10,m)
    u2 = u2 / 10 #pow(10,m)


print("Result: " + str(u))

#ex2

x = 1.0
y = u2/10
z = u2/10
print(x + y)
print(y + z)
print(((x + y) + z) == (x + (y + z)))

x = 1.1
print((x * y) * z)
print(x * (y * z))
print(((x * y) * z) == (x * (y * z)))

#ex3
c1 = 1/factorial(3)
c2 = 1 / factorial(5)
c3 = 1 / factorial(7)
c4 = 1 / factorial(9)
c5 = 1 / factorial(11)
c6 = 1 / factorial(13)
# am pus factorial peste tot in loc de calcul impartit nr anterior pentru ca este mai precis (am testat)
gen = math.pi/2

count = 10000

loss = {f"P{i}": [0, 0, 0] for i in range(1, 9)}
# top3

while count > 0:
    count = count - 1
    num = rand.uniform(-gen, gen)
    xp2 = num * num
    sin = math.sin(num)
    for i in range(1, 9):
        start_time = time.time()
        if i == 1:
            p = num * (1 - xp2 * (c1 - xp2 * c2))
        elif i == 2:
            p = num * (1 - xp2 * (c1 - xp2 * (c2 - xp2 * c3)))
        elif i == 3:
            p = num * (1 - xp2 * (c1 - xp2 * (c2 - xp2 * (c3 - xp2 * c4))))
        elif i == 4:
            p = num * (1 - xp2 * (0.166 - xp2 * (0.00833 - xp2 * (c3 - xp2 * c4))))
        elif i == 5:
            p = num * (1 - xp2 * (0.1666 - xp2 * (0.008333 - xp2 * (c3 - xp2 * c4))))
        elif i == 6:
            p = num * (1 - xp2 * (0.16666 - xp2 * (0.0083333 - xp2 * (c3 - xp2 * c4))))
        elif i == 7:
            p = num * (1 - xp2 * (c1 - xp2 * (c2 - xp2 * (c3 - xp2 * (c4 - xp2 * c5)))))
        elif i == 8:
            p = num * (1 - xp2 * (c1 - xp2 * (c2 - xp2 * (c3 - xp2 * (c4 - xp2 * (c5 - xp2 * c6))))))
        err = abs(sin - p)
        loss[f"P{i}"][0] += err
        loss[f"P{i}"][1] = err
        end_time = time.time()
        loss[f"P{i}"][2] += end_time - start_time


sorted_loss = dict(sorted(loss.items(), key=lambda item: item[1][0]))

print(sorted_loss)


sorted_time = dict(sorted(loss.items(), key=lambda item: item[1][2]))

print(sorted_time)

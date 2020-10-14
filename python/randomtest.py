import random

a = [1,2,3,4,5]
#1
print(random.choices(a,k=3))
#2
print(random.choices(a,weights=[0,0,1,0,0],k=5))
#3
print(random.choices(a,weights=[1,1,1,1,1],k=5))
#4
print(random.choices(a,cum_weights=[1,1,1,1,1],k=5))

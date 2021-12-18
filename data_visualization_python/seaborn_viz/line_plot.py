import matplotlib.pyplot as plt 

list_a  = [1, 2, 3]
list_b  = [1, 4, 9]

list_a  = [i for i in range(100)]
list_b  = [i**2 for i in range(100)]

plt.plot(list_a)
plt.plot(list_b)


plt.show()
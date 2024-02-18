import numpy as np
a = np.array(["Hello", "World"])
a = np.append(a, "!")
print("Current array: {}".format(a))
print("Printing each element")
for i in a:
    print(i)
    
print("\nPrinting each element and their index")
for i,e in enumerate(a):
  print("Index: {}, was: {}".format(i, e))
  
print("\nShowing some basic math on arrays")
b = np.array([0,4,1,2,3])
print("Max: {}".format(np.max(b)))
print("Average: {}".format(np.average(b)))
print("Max index: {}".format(np.argmax(b)))
print("\nYou can print the type of anything")
print("Type of b: {}, Type of 0: {}".format(type(b),type(b[4])))

print("\nUse numpy to create a 3X3 matrix with random numbers")
c = np.random.rand(3,3)
print(c)

print("\nUse numpy to create a 3X3X3 matrix with random numbers")
d = np.random.rand(3,3,3)
print(d)

print("\nPrint the shape of everything")
print("Shape of a: {}".format(a.shape))
print("Shape of b: {}".format(b.shape))
print("Shape of c: {}".format(c.shape))
print("Shape of d: {}".format(d.shape))
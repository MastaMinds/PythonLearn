# Introducing Numpy
import numpy as np

# Tuple and list
MyTuple=(1,2,3,4,5)
MyList=[1.2,3.2,4.5,5.0]
print(type(MyTuple))
print(type(MyList))

# Arrays and operations
A=np.array([1,2,3,4])
B=np.array([7,6,5,4])
C=A+B
print C
E=np.array(MyTuple)
print(type(E))

F=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
print(F)
# Array properties
A.dtype
G=np.eye(3)
print(len(A))
print(G.shape)
print(G.size)
print(G.ndim)

# Change data in an array (Character, complex, float, int)
# Or boolean: True/ False
np.array(A,float)
H=np.array(MyList)
print(np.array(H,complex))

# List of list of lists
I=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])

print(B[0]) # 1st element
print(B[3]) # 4th element
print(F[2,2]) # Element (3,3)

ls=np.linspace(1,10,10)
print(ls)

# Numpy functions
A=np.array(A,float)
print(np.cos(A))
print(A**2)
print(A/2)
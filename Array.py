# Introducing Numpy
from numpy import \
array,linspace,cos,eye,pi,e,exp,sin

#Matplotlib
from matplotlib.pyplot import\
plot,show,figure,subplot,xlim,ylim

# Scipy
import scipy as sc

# Tuple and list
MyTuple=(1,2,3,4,5)
MyList=[1.2,3.2,4.5,5.0]
print(type(MyTuple))
print(type(MyList))

# Arrays and operations
A=array([1,2,3,4])
B=array([7,6,5,4])
C=A+B
print C
E=array(MyTuple)
print(type(E))

F=array([[1,2,3],
            [4,5,6],
            [7,8,9]])
print(F)
# Array properties
A.dtype
G=eye(3)
print(len(A))
print(G.shape)
print(G.size)
print(G.ndim)

# Change data in an array (Character, complex, float, int)
# Or boolean: True/ False
array(A,float)
H=array(MyList)
print(array(H,complex))

# List of list of lists
I=array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])

print(B[0]) # 1st element
print(B[3]) # 4th element
print(F[2,2]) # Element (3,3)

ls=linspace(1,10,10)
print(ls)

# Numpy functions
A=array(A,float)
print(cos(A))
print(A**2)
print(A/2)
print(cos(pi/2))
print([e,exp(1)])

x=linspace(-pi,pi,256,endpoint=True)
C=cos(x)
S=sin(x)

subplot(111)
plot(x,C,color="red",linewidth=1.0,linestyle="-")
plot(x,S,color="blue",linewidth=1.0,linestyle="-")
xlim(-4.0,4.0)
ylim(-1.0,1.0)
show()
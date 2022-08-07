# %% [markdown]
# # NUMPY

# %% [markdown]
# > IMPORT NUMPY LIBRARY

# %%
import numpy as np

# %% [markdown]
# > NUMPY VERSION

# %%
np.__version__

# %% [markdown]
# > DECLARE 1-D ARRAY

# %%
a = np.array([2,4,6,8,10,12,14,16,18,20])
a

# %% [markdown]
# > DECLARE 1-D ARRAY USING RANGE

# %%
b = np.arange(2,41)
b

# %% [markdown]
# > DECLARE ARRAY USING LINESPACE  

# %%
a = np.linspace(2,20,10,dtype=np.int64)
a

# %% [markdown]
# > DECLARE ARRAY HAVING ZERO'S AND ONE'S

# %%
a = np.zeros(5)
a

# %%
a = np.zeros(5,dtype=np.int64)
a

# %%
a = np.ones(5,dtype=np.int32)
a

# %% [markdown]
# > SORT ARRAYS 

# %%
a = np.array([50,41,92,35,9,4,65,850,250,99,100])
print(np.sort(a))

# %% [markdown]
# > SIZE OF ARRAY

# %%
print(np.size(a))

# %% [markdown]
# > APPEND ARRAY

# %%
a=np.append(a,25)
a.sort()
a

# %% [markdown]
# > MATHMATICAL EXPRESSION 
# 
# - SUM
# - MEAN
# - MEDIAN 
# - AVERAGE
# - MAX 
# - MIN

# %%
a.sum()

# %%
a.mean()

# %%
np.median(a)

# %%
int(np.average(a))

# %%
a.max()

# %%
a.min()

# %% [markdown]
# > EXTRACT ELEMENTS FROM ARRAY

# %%
np.extract(a<26,a)

# %% [markdown]
# > DELETE ELEMENTS FROM ARRAY

# %%
np.delete(a,3)

# %% [markdown]
# > CONCATENATE TWO ARRAYS

# %%
a

# %%
b

# %%
np.concatenate((a,b))

# %% [markdown]
# > ARRAY DIMNESION

# %%
a.ndim

# %% [markdown]
# > SHAPE OF ARRAY

# %%
a.shape

# %% [markdown]
# > RESHAPE ARRAY

# %%
a.size
a.reshape(4,3)

# %% [markdown]
# ###### COLUMN VECTOR

# %%
a2 = a [:,np.newaxis]
a2
print(a2)
a2.shape

# %% [markdown]
# ###### ROW VECTOR

# %%
a2 = a [np.newaxis,:]
print(a2)
a2.shape

# %%
a.ndim

# %%
a2.ndim

# %% [markdown]
# > INDEXING 

# %%
a[2]

# %%
a[5:]

# %%
a[-2:]

# %%
a[-5:-2]

# %%
a

# %%
a[a>10]

# %%
a[a%4==0]

# %%
a[(a>10) | (a==18) ]

# %% [markdown]
# > CREATE A AN ARRAY FROM EXISTING DATA

# %%
a

# %%
arr = a[2:7]
arr

# %%
np.hsplit(a,3)

# %% [markdown]
# > ADD TWO ARRAYS

# %%
d = np.array([5,6,7])
e = np.ones(3,dtype=np.int64)
add = d + e
add

# %% [markdown]
# > MULITYPLY ARRAY WITH CONSTANT

# %%
d * 2

# %% [markdown]
# > GENERATE RANDOM ARRAY

# %%
rnd = np.random.default_rng()
rnd


# %%
rnd.random((2,2))

# %%
rnd.integers(5, size=(2,2))

# %%
k = rnd.integers(5, size=(1,20))
k

# %% [markdown]
# > UNIQUE VALUES IN ARRAY

# %%
np.unique(k)

# %% [markdown]
# > REVERSED AN ARRAY

# %%
np.flip(a)

# %% [markdown]
# ---- 

# %% [markdown]
# > DECLARE 2-D ARRAY

# %%
q = np.array([[1,2,3],[4,5,6],[7,8,9]])
q

# %% [markdown]
# > DECLARE 2-D ARRAY USING RANGE AND RESHAPE

# %%
w = np.arange(21,30).reshape(3,3)
w

# %% [markdown]
# > CONCATENATE

# %%
np.concatenate((q,w),axis=0)

# %%
np.concatenate((q,w),axis=1)

# %%
np.concatenate((w,q),axis=0)

# %% [markdown]
# > STACK ARRAY

# %%
np.hstack((q,w))

# %%
np.vstack((q,w))

# %% [markdown]
# > SPLIT ARRAY

# %%
np.split(q,3)

# %%
np.hsplit(q,3)

# %% [markdown]
# > SHAPE 

# %%
q.shape

# %% [markdown]
# > NEWSHAPE (BY USING RESHAPE)

# %%
np.reshape(q,newshape=(1,9),order='F')

# %%
np.reshape(q,newshape=(1,9),order='C')

# %%
np.reshape(q,newshape=(1,9),order='A')

# %% [markdown]
# > DIMENSIONS

# %%
q.ndim

# %% [markdown]
# > SIZE

# %%
q.size

# %% [markdown]
# > FETACH DATA FROM ARRAY

# %%
q[0,0]

# %%
q[0:]

# %%
q[0,0:2]

# %%
q[2,0:2]

# %%
q[q>5]

# %%
q[q%2==0]

# %%
q

# %%
p=np.nonzero(q<6)
p

# %%
list_of_cordinates = list(zip(p[0],p[1]))   ## zip is used to create a list of tuples
for i in list_of_cordinates:
    print(i)

# %% [markdown]
# > CHECK SPECIFIC ELEMENT PRESENT IN 2-D ARRAY OR NOT

# %%
check = np.nonzero(q == 6)
h = list(zip(check[0],check[1]))
if h != []:
    print("Element found at",h)
else:
    print("Element not found")

# %%
check = np.nonzero(q == 16)
h = list(zip(check[0],check[1]))
if h != []:
    print("Element found at",h)
else:
    print("Element not found")

# %% [markdown]
# > MATHMATICAL EXPRESSION 
# 
# - SUM
# - MEAN
# - MEDIAN 
# - AVERAGE
# - MAX 
# - MIN

# %%
q.sum()

# %%
q.mean()

# %%
int(np.median(q))

# %%
int(np.average(q))

# %%
q.max()

# %%
q.max(axis=0)

# %%
q.max(axis=1)

# %%
q.min()

# %%
q.min(axis=0)

# %%
q.min(axis=1)

# %% [markdown]
# > ADD TWO MATRICS

# %%
o =np.array([[1,1,1],[1,1,1],[1,1,1]])
add = q + o
add

# %%
o = np.array([[1,1,1]])
add = q + o
add

# %% [markdown]
# > MULTIPLY TWO MATRICS

# %%
Multiply = q * add
Multiply

# %% [markdown]
# > APPEND IN ARRAY

# %%
u = np.append(q,[[1,2,3]],axis=0)
u

# %% [markdown]
# CHECK UNIQUE ELENMENTS

# %%
 np.unique(u)

# %%
t = np.unique(u,return_counts=True)
t

# %% [markdown]
# > TRANSPOSE

# %%
q.transpose()

# %%
q.T

# %%
q

# %% [markdown]
# > REVERSED ARRAY

# %%
np.flip(q)

# %% [markdown]
# ###### FLIP SPECIFIC ROW OR COLUMN

# %%
arr = np.copy(q)
arr[:,1]= np.flip(q[:,1])
print(arr)

# %% [markdown]
# >  CONVERT 2-D ARRAY IN TO 1-D

# %%
n= arr.flatten()
n

# %%

q

# %%
g = np.copy(q)
print(g)
n = np.ravel(g)
print(n)
print(g)

# %% [markdown]
# -----

# %% [markdown]
# > DECLARE 3-D ARRAY

# %%
d = np.array([[[1,2],[5,4]],
            [[8,7],[9,3]],
            [[11,5],[6,12]]
            ])
d

# %% [markdown]
# > DIMENSION OF ARRAY

# %%
d.ndim

# %% [markdown]
# >SIZE OF ARRAY

# %%
d.size

# %% [markdown]
# > INDEXING

# %%
d[2,0,0 ]

# %%
d[0,1,0 :]

# %%
d[0,-1,0]

# %%
d[1,:,:]

# %% [markdown]
# > SHAPE OF ARRAY

# %%
d.shape

# %% [markdown]
# > RESHAPE ARRAY

# %%
d.reshape(12)

# %% [markdown]
# > ITERATING OF EACH SCALAR ELEMENT

# %%
for i in np.nditer(d):
    print(i)

# %%
d

# %%
e = np.array([[[10,20],[40,50]],[[70,80],[90,100]]])
e

# %% [markdown]
# > CONCATENATE OR JOIN ARRAYS

# %%

np.concatenate((d,e),axis=0)

# %% [markdown]
# > APPEND 3-D ARRAY

# %%
f =np.array([[[15,30],[25,45]]])
e =np.append(e,f,axis=0)
e

# %%
np.hstack((d,e))

# %%
np.vstack((d,e))

# %% [markdown]
# > TRANSPOSE

# %%
e.T

# %% [markdown]
# > SUM

# %%
np.sum(d)

# %%
d

# %%
print(np.shape(d),d.ndim)

# %% [markdown]
# > ROTATE 3D ARRAY 90 DEGREE

# %%
h=np.rot90(d)
print(h)
print(np.shape(h),h.ndim)

# %% [markdown]
# > FIND ELEMENT USING WHERE METHOD

# %%
result = np.where(d>7)
result



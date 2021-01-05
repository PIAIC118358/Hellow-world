#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[5]:


import numpy as np


# 2. Create a null vector of size 10 

# In[63]:


a = np.zeros(10)
a


# 3. Create a vector with values ranging from 10 to 49

# In[64]:


b = np.arange(10,50)
b


# 4. Find the shape of previous array in question 3

# In[65]:


np.shape(b)


# 5. Print the type of the previous array in question 3

# In[66]:


type(b)


# 6. Print the numpy version and the configuration
# 

# In[67]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[68]:


b.shape


# 8. Create a boolean array with all the True values

# In[69]:



np.ones(5, dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[70]:


two_d_a=np.zeros((2,2))
print(two_d_a)


# 10. Create a three dimensional array
# 
# 

# In[71]:


three_d_a=np.zeros((3,3,3))
print(three_d_a)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[72]:


x = np.arange(1, 10)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[73]:


x = np.zeros(10)
print(x)
print("Update fiftha value to 1")
x[5] = 1
print(x)


# 13. Create a 3x3 identity matrix

# In[74]:


x=np.identity(3)
print('3x3 matrix:')
print(x)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[81]:


arr = np.array([1, 2, 3, 4, 5], dtype=float)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[82]:



arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
x=arr1*arr2
x


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[6]:


arr1 = np.array([1., 2., 3.]) 
arr2 = np.array([0., 4., 1.])
arr1 < arr2


# 17. Extract all odd numbers from arr with values(0-9)

# In[9]:


a = np.array([1,2,3,4,5,6,6,7,8,9])
a[a % 2 == 1]


# 18. Replace all odd numbers to -1 from previous array

# In[37]:


v1 = np.array([1,2,3,4,5,6,6,7,8,9]) 
v1[v1%2==1] *= -1          

print (v1)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[39]:


arr = np.arange(10)
arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[32]:


x = np.ones((5,5))
x[1:-1,1:-1] = 0
print(x)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[27]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1][1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[30]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0][0]=64
arr3d[0][1]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[70]:


data = np.array([[1, 2, 3],
                 [4, 5, 6],
                [7, 8, 9]])
X = data[0:, :]
print(X)


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[66]:


data = np.array([[1, 2, 3],
                 [4, 5, 6],
                [7, 8, 9]])
X = data[1:, :]
print(X)


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[87]:


data = np.array([[1, 2, 3],
                 [4, 5, 6],
                [7, 8, 9]])
X = data[0:, 0:1]
print(X)


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[5]:



x = np.random.random((10,10))
print("Original Array:")
print(x) 
xmin, xmax = x.min(), x.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[22]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b)) 


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[10]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[28]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
np.in1d(names, data)


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[31]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
np.setdiff1d(names, data)


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[34]:


a=np.random.rand(1,15).reshape(5,3)
a


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[35]:


a=np.random.rand(1,16).reshape(2,2,4)
a


# 33. Swap axes of the array you created in Question 32

# In[36]:


a=np.random.rand(1,16).reshape(2,2,4)
np.swapaxes(a,0,1)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[37]:


a=np.arange(10)
a=np.sqrt(a)
np.where(a<0.5,0,a)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[44]:


a=[1,2,3,4,5]
b=[2,3,4,6,7]
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[45]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[49]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
for i, val in enumerate(a):
    if val in b:
        a = np.delete(a, np.where(a == val)[0][0])
        b = np.delete(b, np.where(b == val)[0][0])

for i, val in enumerate(b):
    if val in a:
        a = np.delete(a, np.where(a == val)[0][0])
        b = np.delete(b, np.where(b == val)[0][0])

print(a)
print(b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[54]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray = np.delete(sampleArray, 1, axis=1)
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[55]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[62]:


a=np.random.rand(20)
np.cumsum(a)


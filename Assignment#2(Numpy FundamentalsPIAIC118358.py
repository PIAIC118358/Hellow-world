#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[20]:


A = np.array([1,2,3,4,5,6])
B = np.reshape(A, (-1, 3))
B


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[17]:


a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
b = np.array([[1,1,1,1,1],[1,1,1,1,1]])
np.vstack((a,b))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[21]:


a = np.array([[0,1,2,3,4],[5,6,7,8,9]])
b = np.array([[1,1,1,1,1],[1,1,1,1,1]])
np.hstack((a,b))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[23]:


a = np.array([[0,1,2,3,4], [5,6,7,8,9]])

b = a.ravel()

b


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[39]:


a=np.array([[ 0,  1,  2,  3, 4],
            [ 5,  6,  7, 8, 9],
            [ 10, 11, 12, 13, 14]])
a.ravel()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[50]:


a=np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
b = a.reshape(1, 5, 3)

b


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[63]:


a = np.random.random((5,5))
b=np.square(a)
b


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[64]:


a = np.random.random((5,6))
b=np.mean(a)
b


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[65]:


a = np.random.random((5,6))
np.std(a)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[66]:


a = np.random.random((5,6))
np.median(a)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[70]:


a = np.random.random((5,6))
b=np.transpose(a)
b


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[77]:


n_array = np.array([[2, 2, 2, 2], 
                    [4, 4, 4, 4], 
                    [6, 6, 6, 6],
                    [8, 8, 8, 8]]) 
diag = np.diagonal(n_array) 
sum(diag)


# ## Question:13

# # Find the determinant of the previous array in Q12?

# In[79]:


a = np.array([[2, 2, 2, 2], 
                    [4, 4, 4, 4], 
                    [6, 6, 6, 6],
                    [8, 8, 8, 8]]) 
np.linalg.det(a)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[84]:


a = np.array([[2, 2, 2, 2], 
                    [4, 4, 4, 4], 
                    [6, 6, 6, 6],
                    [8, 8, 8, 8]]) 
b=np.percentile(a, 5)

b


# ## Question:15

# ### How to find if a given array has any null values?

# In[8]:


a = np.array(range(5,20,2))
a

b = np.append(a,np.nan)
b

c=np.isnan(a)
prin


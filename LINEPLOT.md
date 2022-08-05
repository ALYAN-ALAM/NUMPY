# LINE PLOT


```python
import seaborn as sns
import matplotlib.pyplot as plt

iris =sns.load_dataset("iris")
# print(iris)

p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris)
plt.show()
```


    
![png](output_1_0.png)
    


## - change figure size
\


```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris =sns.load_dataset("iris")
plt.figure(figsize=(10,8))

p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris)

plt.show()
```


    
![png](output_3_0.png)
    


## - change color


```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris =sns.load_dataset("iris")
plt.figure(figsize=(10,8))

p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,color="grey")

plt.show()
```


    
![png](output_5_0.png)
    


## - change palette (color set)


```python
import seaborn as sns
import matplotlib.pyplot as plt

iris =sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species")

plt.show()
```


    
![png](output_7_0.png)
    


## -change style 
- darkgrid
- whitegrid
- dark
- white 
- ticks


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species")
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_9_0.png)
    


## - line style 
- dotted
- dashed
- dashdot
- solid


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="dotted")
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_11_0.png)
    


## - change line width


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="dashed",linewidth=3)
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_13_0.png)
    


## - change drawstyle 


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(10,8))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="solid",linewidth=2,drawstyle="steps-pre")
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_15_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(10,8))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="solid",linewidth=2,drawstyle="steps-mid")
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_16_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(10,8))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="solid",linewidth=2,drawstyle="steps-post")
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_17_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
iris =sns.load_dataset("iris")
plt.figure(figsize=(8,6))
sns.set_palette("husl")
p = sns.lineplot(x="sepal_length",y="sepal_width",data=iris,hue="species",linestyle="solid",linewidth=2
                )
plt.title("LINE PLOT")
plt.show()
```


    
![png](output_18_0.png)
    


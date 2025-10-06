# all the imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import StratifiedShuffleSplit

#Step 1
data = pd.read_csv("Project 1 Data.csv")

print(data.head(), "\n")
print(data.columns, "\n")

#Step 2

#scatter plots of x, y and z vs. step to understand behaviour withing each class.

#the bar chart shows that the dataset is imbalanced. Some steps have many more 
#samples than others. This means models could become biased towards the majority 
#steps
x = data["X"].to_numpy()
y = data["Y"].to_numpy()
z = data["Z"].to_numpy()
step = data["Step"].to_numpy()

plt.scatter(x, step, alpha=0.5)
plt.title("X vs Step")
plt.xlabel("X")
plt.ylabel("Step")
plt.show()

plt.scatter(y, step, alpha=0.5)
plt.title("Y vs Step")
plt.xlabel("Y")
plt.ylabel("Step")
plt.show()

plt.scatter(z, step, alpha=0.5)
plt.title("Z vs Step")
plt.xlabel("Z")
plt.ylabel("Step")
plt.show()

unique_steps, counts = np.unique(step, return_counts=True)
plt.bar(unique_steps, counts)
plt.title("Number of samples per Step")
plt.xlabel("Step")
plt.ylabel("Count")
plt.show()

#visualizing the count of each step.

#findings show that most of the steps are centered around 7,8,9, meaning that it 
#will lead to bias and requiring addtional optimization.

#Step 3
corr = data[["X", "Y", "Z", "Step"]].corr(method="pearson")
print("Corr Matrix:\n", corr.round(4), "\n")

sb.heatmap(corr, annot=True, cmap="Reds", vmin= -1)
plt.title("correlation matrix")

# X has the strongest negative corrilation with the step, moderately with Y and 
# weakly with Z This means X will have the greatest impact on the model’s 
# predictions, while Y and Z have smaller positive effects. 

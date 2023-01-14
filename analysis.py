import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#split the data to 5 tables to see the corr

part1 = pd.read_csv('part1.csv')
part2 = pd.read_csv('part2.csv')
part3 = pd.read_csv('part3.csv')
part4 = pd.read_csv('part4.csv')
part5 = pd.read_csv('part5.csv')

# sns.heatmap(part1.corr())
# sns.heatmap(part2.corr())
# sns.heatmap(part3.corr())
# sns.heatmap(part4.corr())
# sns.heatmap(part5.corr())
sns.pairplot(part2)
plt.show()
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

pd.options.display.max_columns = None
pd.options.display.max_rows = None

start_time=time.time()
df_encoded = pd.DataFrame()

# Create a LabelEncoder object
le = LabelEncoder()
# Load the dataset
train = pd.read_csv("train.csv")
# Check the type of data stored in each column

################################################print(train.info())

# Fill in missing values for the LotFrontage and MasVnrArea features
obj_median = ["LotFrontage","MasVnrArea"]

for i in obj_median:
    train[i].fillna(train[i].median(), inplace=True)

train["GarageYrBlt"] = train["GarageYrBlt"].combine_first(train["YearBuilt"])

#print(train.info())

# Drop irrelevant continuous variable
train.drop(["Id"], axis=1, inplace=True)
train.drop(["Alley"], axis=1, inplace=True)
train.drop(["PoolQC"], axis=1, inplace=True)
train.drop(["MiscFeature"], axis=1, inplace=True)
tr_cont = train.copy()

#check where missing data
for i in tr_cont.columns:
    if tr_cont[i].isna().sum() > 0:
        pass
#       print(i,tr_cont[i].isna().sum(),tr_cont[i].dtype)

#all the missing data type is object and it will be encoding

# takes a column and encodes the strings to numerical values
for i in tr_cont.columns:
    if tr_cont[i].dtype=="object":
        le.fit(tr_cont[i])
        tr_cont[i]=le.transform(tr_cont[i])


# Converting dtype to 'int64'

for i in tr_cont.columns:
    if tr_cont[i].dtype != "int64":
        tr_cont[i] = tr_cont[i].astype("int64")

#after cleaning data lets run the model
X = tr_cont.drop("SalePrice", axis=1)
y = tr_cont["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the DecisionTreeRegressor class
reg = DecisionTreeRegressor()

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)


end_time=time.time()
# Evaluate the model's performance

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("R-Squared: ", r2)
print("Training time: {:.4f} seconds".format(end_time-start_time))
depth = reg.tree_.max_depth
print("Maximum depth of the tree: ", depth)


plt.figure(figsize=(50,80))
plot_tree(reg, filled=True, max_depth=4);
plt.show()

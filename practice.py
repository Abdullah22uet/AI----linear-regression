# we use numpy library for using concat,sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# this library for splitting data into training and testing
from sklearn.model_selection import train_test_split
# this library for model training
from sklearn.linear_model import LinearRegression
# this library for finding mean square error
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r"D:\abdullah\abbbb python\aapandas\AI -- linear regression\mpg.csv")

# plot heatmap to check null values in the dataset
sns.heatmap(df.isnull() , yticklabels=False , cbar=False , cmap="tab20c_r")
plt.title("Missing data")
plt.show()

# clean the dataset by removing duplicate and null values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop(["origin","name"],inplace=True,axis=1)

# change float datatype into integer
o = ["mpg","displacement","horsepower","acceleration"]
for item in o:
    df[item] = df[item].astype("int")

x = df.drop("weight",axis=1)
y = df["weight"]

# splitting data into training and testing part
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=101)

# feeding data into model 
model = LinearRegression()
model.fit(x_train,y_train)

# now testing the model
out = model.predict(x_test)

# see results by comparing
result = np.column_stack((y_test,out))
print("Actual values | Predicted values")
print("--"*40)
for actual,predicted in result:
    print(f"{actual:7.1f} | {predicted:7.1f}")

# see results by differentiation
residual = actual-out.reshape(-1)
print(residual)

# finding mean square error
print("Mean square error")
print("--"*40)
mse = mean_squared_error(y_test,out)
rmse = np.sqrt(mse)
print("Mean error : ",mse)
print("Root : ",rmse)

# ploting linear regression line between testing and predicting data
sns.scatterplot(x=y_test, y=out, color='blue', label='Actual Data points')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual (Linear Regression)')
plt.legend()
plt.show()
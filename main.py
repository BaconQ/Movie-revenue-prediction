import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


dataset = pandas.read_csv('cost_revenue_clean.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

plt.figure(figsize=(10,6))
plt.scatter(x,y, alpha=0.3)
plt.title("Film Cost vs Global Revenue")
plt.xlabel("Production Budget $")
plt.ylabel("worldwide Gross $")
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)

regression = LinearRegression()
regression.fit(x,y)
slope = regression.coef_
y_intercept = regression.intercept_
plt.plot(x, regression.predict(x), color="red", linewidth=4)
print(regression.score(x,y))
plt.show()


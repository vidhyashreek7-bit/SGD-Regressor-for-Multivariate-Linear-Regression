# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize dataset with input features and target outputs.

2.Apply feature scaling to normalize the data.

3.Train the SGD Regressor model using the scaled data.

4.Predict outputs for given input using the trained model.

5.Display predicted values and plot Actual vs Predicted graph.
## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VIDHYA SHREE K
RegisterNumber:  212225230296

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
X = np.array([
    [1000, 2],
    [1500, 3],
    [1800, 4],
    [2400, 3],
    [3000, 5]
])

y = np.array([
    [200000, 3],
    [300000, 4],
    [350000, 5],
    [450000, 4],
    [600000, 6]
])

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

models = []
for i in range(y_scaled.shape[1]):
    model = SGDRegressor(max_iter=2000, eta0=0.01)
    model.fit(X_scaled, y_scaled[:, i])
    models.append(model)

y_pred_scaled = np.column_stack([m.predict(X_scaled) for m in models])
y_pred = scaler_y.inverse_transform(y_pred_scaled)


plt.figure()

plt.scatter(y[:,0], y_pred[:,0])

min_val = min(y[:,0].min(), y_pred[:,0].min())
max_val = max(y[:,0].max(), y_pred[:,0].max())
plt.plot([min_val, max_val], [min_val, max_val])  # y = x line

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (SGD Regressor)")

plt.show()
new_data = np.array([[2000, 3]])
new_scaled = scaler_X.transform(new_data)

pred_scaled = [m.predict(new_scaled)[0] for m in models]
pred = scaler_y.inverse_transform([pred_scaled])

print("Predicted Price:", pred[0][0])
print("Predicted Occupants:", pred[0][1])

```

## Output:
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/73054bba-d141-405c-b5f9-760f1c33dd3a" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

np.random.seed(42)

n_samples = 500

Project_Size = np.random.randint(50,500,n_samples)
Team_Size = np.random.randint(3,15,n_samples)
Experience_Level = np.random.randint(1,10,n_samples)

# corrected effort formula with decreasing effort for larger teams and more experience
Effort = (
    0.8 * Project_Size
    - 5 * Team_Size
    - 8 * Experience_Level
    + 100
    + np.random.normal(0, 10, n_samples)
)

X = np.column_stack((Project_Size,Team_Size,Experience_Level))
y = Effort

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

class ELM:

    def __init__(self,hidden=20):
        self.hidden = hidden

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def fit(self,X,y):

        features = X.shape[1]

        self.W = np.random.randn(features,self.hidden)
        self.b = np.random.randn(self.hidden)

        H = self.sigmoid(X @ self.W + self.b)

        self.beta = np.linalg.pinv(H) @ y

    def predict(self,X):

        H = self.sigmoid(X @ self.W + self.b)

        return H @ self.beta


elm = ELM(20)
elm.fit(X_train,y_train)

lr = LinearRegression()
lr.fit(X_train,y_train)

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)

svm = SVR()
svm.fit(X_train,y_train)


def predict_effort(project_size,team_size,experience):

    data = np.array([[project_size,team_size,experience]])

    data_scaled = scaler.transform(data)

    prediction = elm.predict(data_scaled)

    return max(0, round(float(prediction[0]),2))


def get_comparison_results():

    results = {}

    results["ELM"] = mean_absolute_error(y_test,elm.predict(X_test))
    results["Linear Regression"] = mean_absolute_error(y_test,lr.predict(X_test))
    results["KNN"] = mean_absolute_error(y_test,knn.predict(X_test))
    results["SVM"] = mean_absolute_error(y_test,svm.predict(X_test))

    return results
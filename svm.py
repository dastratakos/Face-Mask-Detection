from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_train = [[0, 0], [1, 1]]
Y_train = [0, 1]

X_test = [[-1, -1]]
Y_test = [1]

linearclf = make_pipeline(StandardScaler(), SVC())
linearclf.fit(X_train, Y_train)

predictions = linearclf.predict(X_test)
print(predictions)
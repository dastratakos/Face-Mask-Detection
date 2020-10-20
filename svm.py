from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_train = [[0, 0, 0], [1, 1, 1]]
Y_train = [0, 1]

X_val = [[-1, -1, -1]]
Y_val = [1]

linearclf = make_pipeline(StandardScaler(), SVC())
linearclf.fit(X_train, Y_train)

predictions = linearclf.predict(X_val)
print(predictions)
print("Score is", linearclf.score(X_val, Y_val))
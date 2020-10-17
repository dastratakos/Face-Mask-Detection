from sklearn import svm

X_train = [[0, 0], [1, 1]]
Y_train = [0, 1]

X_test = [[-1, -1]]
Y_test = [1]

linearclf = svm.SVC()
linearclf.fit(X_train, Y_train)

predictions = linearclf.predict(X_test)
print(predictions)
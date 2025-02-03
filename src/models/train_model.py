from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle



def train_logistic_regression(xtrain, ytrain):

    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    with open('models/LRmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


def train_random_forest(xtrain, ytrain, n_estimators=100, max_depth=3, max_features='auto'):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    model.fit(xtrain, ytrain)

    with open('models/RFmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model
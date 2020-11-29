import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class CancerPrediction:
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        cell_df = pd.read_csv("cell_samples.csv")
        return cell_df

    def data_processing(self):
        cell_df = self.load_data()
        cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
        cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
        return cell_df

    def create_feature(self):
        cell_df = self.data_processing()
        feature_df = cell_df[
            ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
        X = np.asarray(feature_df)
        cell_df['Class'] = cell_df['Class'].astype('int')
        y= np.asarray(cell_df['Class'])
        self.train_test_data(X, y)

    def train_test_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        print('Train Data', X_train.shape, y_train.shape)
        print('Test Data', X_test.shape, y_test.shape)
        self.hyperparameter_tuning(X_train, X_test, y_train, y_test)

    def hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # No of features
        max_features = ['auto', 'sqrt', 'log2']
        # Max depth for a RF
        max_depth = [int(x) for x in np.linspace(10, 1000, 10)]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 14]
        # min sample in each leaf node
        min_samples_leaf = [1, 2, 4, 6, 8]
        # Create a dictonary of a random_grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'criterion': ['entropy', 'gini']}
        print(random_grid)
        rf = RandomForestClassifier()
        rf_randomcv = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                         random_state=100, n_jobs=1)
        rf_randomcv.fit(X_train, y_train)
        best_grid = rf_randomcv.best_estimator_
        best_grid
        self.metrics(self, best_grid, X_test, y_test)

    @staticmethod
    def metrics(self, best_grid, X_test,y_test):
        y_pred = best_grid.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print('accuracy', accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


def main():
    obj=CancerPrediction()
    obj.create_feature()


if __name__ == "__main__":
    main()

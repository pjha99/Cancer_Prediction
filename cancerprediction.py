import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class CancerPrediction:
    """
    Cancer Prediction class in which all the operations are performed.
    From loading the data to train test split and applying random forest method
    and then hyper tuning of the models
    """

    def __init__(self):
        pass

    @staticmethod
    def load_data():
        """load the cancer data
           Return: Cell_df
           """

        cell_df = pd.read_csv("cell_samples.csv")
        return cell_df

    def data_processing(self):
        """
        This functions performs the data pre processing
        """

        cell_df = self.load_data()
        cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
        cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
        return cell_df

    def create_feature(self):
        """
        Here we select the independent variable that we want to use
        """

        cell_df = self.data_processing()
        feature_df = cell_df[
            ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
        X = np.asarray(feature_df)
        cell_df['Class'] = cell_df['Class'].astype('int')
        y= np.asarray(cell_df['Class'])
        self.train_test_data(X, y)

    def train_test_data(self, X, y):
        """We split our data into train and test data
         Train Data : 80%
         Test Data :20 %
         """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
        print('Train Data', X_train.shape, y_train.shape)
        print('Test Data', X_test.shape, y_test.shape)
        self.randomized_search(X_train, X_test, y_train, y_test)

    def randomized_search(self, X_train, X_test, y_train, y_test):
        """
        Here we Performe hyper parameter tuning of randomized search cv on the random forest
        the best grid is selected which acts as base params for the
        grid search cv
        """
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
        print('Results after Randomized Search')
        self.metrics(self, best_grid, X_test, y_test)
        print('Now Calling Grid search')
        best_grid_1=self.grid_search(rf_randomcv.best_estimator_,X_train,y_train)
        print('Results after Grid search CV')
        self.metrics(self, best_grid_1, X_test, y_test)

    def grid_search(self, best_grid,X_train, y_train):
        """
        Here we take the best param from the randomized search cv
        and apply grid searcg cv on Random Forest algorithm
        """
        param_grid_1 = {
            'criterion': [best_grid['criterion']],
            'max_depth': [best_grid['max_depth']],
            'max_features': [best_grid['max_features']],
            'min_samples_leaf': [best_grid['min_samples_leaf'],
                                 best_grid['min_samples_leaf'] + 2,
                                 best_grid['min_samples_leaf'] + 4],
            'min_samples_split': [best_grid['min_samples_split'] - 2,
                                  best_grid['min_samples_split'] - 1,
                                  best_grid['min_samples_split'],
                                  best_grid['min_samples_split'] + 1,
                                  best_grid['min_samples_split'] + 2],
            'n_estimators': [best_grid['n_estimators'] - 200, best_grid['n_estimators'] - 100,
                             best_grid['n_estimators'],
                             best_grid['n_estimators'] + 100, best_grid['n_estimators'] + 200]
        }
        param_grid_1
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_1, cv=10, verbose=2)
        grid_search.fit(X_train, y_train)
        best_grid_1=grid_search.best_estimator_
        print(best_grid_1)
        return best_grid_1

    @staticmethod
    def metrics(self, best_grid, X_test,y_test):
        """
        Here we calculate final metrics for our classification performed
        the various metrics that are used are:
        1.)Accuracy
        2.)Precision
        3.)Recall
        4.)F1-score"
        """
        y_pred = best_grid.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print('accuracy', accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))


def main():
    obj=CancerPrediction()
    obj.create_feature()


if __name__ == "__main__":
    main()

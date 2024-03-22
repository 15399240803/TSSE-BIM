# TSSE-BIM
Example:
  ```
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from utils import make_binary_classification_target, imbalance_train_test_split
    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, True)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
            X, y, test_size=0.2, random_state=42)
    tsse = TSSE(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )
    print('auc_prc_score: {}'.format(tsse.score(X_test, y_test)))
 ```

If you need to modify the boundary undersampling weights, modify lines 133 through 145 of tsse.py.
![image](https://github.com/15399240803/TSSE-BIM/assets/63033258/204eb574-a5e5-4124-aca4-afa105eb7720)

The weights of the dataset creditcard are [1,2,5,30,20].

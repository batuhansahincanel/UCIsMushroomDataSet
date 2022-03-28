#UCIsMushroomDataSet

This is a non-scientific Artificial Neural Networks study on UCI's Mushroom Dataset.

Source;

https://archive.ics.uci.edu/ml/datasets/mushroom

In the dataset;
- There are 22 independent variables and all are categorical form.
- The dependent variable is consisted of 2 classes. (p for poisoned and e for edible)
- There are 2480 missing attiributes and all in the same column.


In this study;
- The column which contains all the missing attiributes is dropped.
- The independent variables are encoded by frequency map method beside One Hot Encoding.
- The independent variables get scaled by standard scaler.
- The dependent variable is label encoded. (1 for poisoned and 0 for edible)
- The Artificial Neural Network contains 2 hidden layer.
- The success of this model in estimating whether the mushroom is poisonous or edible is %99 which is measured by accuracy score and confusion matrix.

# using wikipedia data to identify iris flowers
# https://en.wikipedia.org/wiki/Iris_flower_data_set?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt
# import dataset - train classifier - predict label for new flower - visualize the tree

# from sklearn.datasets import load_iris

#import dataset from http://scikit-learn.org/stable/datasets/?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt

# iris = load_iris()

# print iris.feature_names
# print iris.target_names
# print iris.data[0]
# print iris.target[0]

# To print out all data iterativly we do the following
# This is a way to test that you've got access and understanding of external data

# for i in range (len(iris.target)):
#     print "Example%d: label %s, features %s" % (i, iris.target[i], iris.data[i])

# Now we want to train a classifer
# Moving aside testing data to check classifiers's accuracy

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# visualize the tree
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

from sklearn import tree

# Supervised Learning
# - collect training data
#   - Classify behaviour by features
#   - Good features provide easy, effective discrimination
#   - All the examples of data we want to learn from
#   -
# -

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Decision Tree for a classifier (box of rules)

clf = tree.DecisionTreeClassifier()

# To train Decision Tree we need a learning algorithm

clf = clf.fit(features, labels)

print clf.predict([[160, 0]])

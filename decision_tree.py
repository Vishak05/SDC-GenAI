import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Load Titanic dataset from seaborn
df = sns.load_dataset('titanic')

# Step 2: Preprocessing
# Drop rows with missing values in selected columns
df = df[['pclass', 'sex', 'age', 'fare', 'survived']].dropna()

# Convert categorical to numerical
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Step 3: Features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4)
clf.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Step 7: Visualize the decision tree
plt.figure(figsize=(18, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=["Died", "Survived"], filled=True)
plt.title("Titanic Survival Prediction Tree")
plt.show()

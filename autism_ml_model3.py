import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import plotly.express as px
import seaborn as sns


autism_df = pd.read_csv("toddlersnormalizedbalance1.csv")
autism_df.dropna(inplace=True)

output = autism_df['ASD']
features = autism_df[['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']]

#print(output.tail())
#print(features.tail())
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.3, random_state=0)

def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(), labels = {'x':'Importance Value', 'index':'QCHAT Screening Questions (10)'},
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Importance Plot',
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)

# Get feature importances
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(x_train, y_train)
feat_importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)
impPlot(feat_importances, 'Random Forest Classifier')
st.write('\n')
y_pred = rf.predict(x_test)
score = accuracy_score(y_pred, y_test)
report = classification_report(y_test, y_pred)
print("Our accuracy score for this model feature importance is {}".format(score))
st.text('Classification Report:\n\n{}'.format(report))

# Perform RFE with cross-validation
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rfc_selector = RFE(rfc, step=1)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
pipeline = Pipeline(steps=[('s',rfc_selector), ('m', rfc)])
n_features = [2, 4, 6, 8] # Define the number of features to select
rfe_results = []
for i in n_features:
    rfc_selector.set_params(n_features_to_select=i)
    results = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring='accuracy')
    rfe_results.append(results)
    st.write('Number of features selected:', i, 'Accuracy:', results.mean())
    
# Select the optimal number of features
optimal_n_features = n_features[np.argmax(np.mean(rfe_results, axis=1))]
st.write('Optimal number of features:', optimal_n_features)


# Get the selected features using the optimal number of features
rfc_selector.set_params(n_features_to_select=optimal_n_features)
rfc_selector.fit(x_train, y_train)
selected_features = features.columns[rfc_selector.support_]

# Print the selected features
st.write("Selected features:", selected_features)

import matplotlib.pyplot as plt

# Get the predicted probabilities for the test set
y_pred_proba = rf.predict_proba(x_test)[:,1]

# Compute the false positive rate and true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)

# Compute the area under the curve (AUC)
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC curve
fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
ax_roc.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc="lower right")

# Plot the confusion matrix
fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix')

# Show the plots in Streamlit
st.pyplot(fig_roc)
st.pyplot(fig_cm)


#Save the model and output encodings to pickle files
with open("random_forest_autism3.pickle", 'wb') as rfc_pickle:
    pickle.dump(optimal_n_features, rfc_pickle)

with open('output_autism3.pickle', 'wb') as output_pickle:
    pickle.dump(uniques, output_pickle)




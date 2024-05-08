#importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc
warnings.filterwarnings("ignore")
#reading the dataset
data=pd.read_csv("credit.csv")
data.head()
data["default"].value_counts()
#Understanding the dataset- number of rows, columns, types of columns etc
data.info()
#There are 1000 rows in the dataset and no null value is present, so describing the summary statistics of the numerical columns data and checking for duplicate rows
print(data.duplicated().sum())
data.describe()
#Summarizing the statistics of all the  categorical columns
data.describe(include=["O"])
#Now diving deep into the occurence of each value in each category of categorical columns, to understand the frequency distribution of each value
categorical_columns=[col for col in data.columns if (data[col].dtype=="object")]
print(categorical_columns)
lis=[]
for col in categorical_columns:
    categories=data[col].unique()
    for item in categories:
        freq=(data[col]==item).sum()
        lis.append({"Feature":col,"Value":item,"Count":freq})
x=pd.DataFrame(lis)
x.head(n=39)
#There is one category in purpose columns called car0, which seems to be a typing mistake so correcting it to count its frequencies under "car" only.
data["purpose"]=data["purpose"].str.replace("car0","car")
#Summarizing the statistics of all the  categorical columns(after converting car0 to car)
data.describe(include=["O"])
#Splitting the columns into numerical columns and categorical columns for visualizing with different plots
cont_cols=["months_loan_duration","amount","age","percent_of_income","years_at_residence","existing_loans_count","dependents"]
disc_cols=["checking_balance","credit_history","purpose","savings_balance","employment_duration","other_credit","housing","job","phone"]
#Histogram plots for numerical columns for  understanding the distribution
for col in cont_cols:
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
#Boxplots for numerical columns with "default" column as hue 
for cont in cont_cols:
    sns.boxplot(data=data, x=cont, y="default")
    plt.show()
#Pairplot for numerical columns with "default" column as hue to understand the relationship between them
sns.pairplot(data,hue="default",vars=cont_cols)
#Correlation matrix and heatmap between numerical columns
correlation_matrix=(data[cont_cols]).corr()
sns.heatmap(correlation_matrix)
correlation_matrix
#Comparing the categorical column counts using a countplot with "default" as the hue parameter
for col in disc_cols:
    sns.countplot(data=data, x=col,hue="default")
    plt.title(f'Countplot for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
#Plots for understanding the relationship between categorical columns one at a time
for disc1 in disc_cols:
    for disc2 in disc_cols:
        if disc1 != disc2: 
            sns.countplot(data=data, x=disc1, hue=disc2)
            plt.title(f'Countplot for {disc1} vs {disc2}')
            plt.xlabel(disc1)
            plt.ylabel('Count')
            plt.legend(title=disc2)
            plt.show()
#Modelling
#Now dicrete columns can either be nominal or ordinal so encoding accordingly
#Encoding the ordinal columns with OrdinalEncoder
orders = {'credit_history': ['critical','poor', "good", "very good", 'perfect'],'employment_duration': ['unemployed', '< 1 year', '1 - 4 years','4 - 7 years','> 7 years']}
ordinal_data = pd.DataFrame()
for col,order in orders.items():
    sorted_order = sorted(order)
    encoder = OrdinalEncoder(categories=[sorted_order])
    encoded_column = encoder.fit_transform(data[[col]])
    ordinal_data[col+'_encoded']=encoded_column.ravel()
ordinal_data.head()
#List of categorical columns except the ordinal ones(the nominal columns)
one_hot_cols=[col for col in disc_cols if col not in orders.keys()]
one_hot_cols
#Encoding the nominal columns using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data[one_hot_cols])
encoded_df= pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(one_hot_cols))
encoded_df.shape
encoded_df.head()
#Concatenating all the numerical, ordinal and nominal columns into a single dataframe after encoding
final_dataframe=pd.concat([data,encoded_df,ordinal_data],axis=1)
final_dataframe.head(n=20)
#Dropping the original dicrete columns as we have encoded columns now
final_dataframe.drop(columns=one_hot_cols,inplace=True)
final_dataframe.drop(columns=list(orders.keys()),inplace=True)
#Mapping the target column "default" to 0 and 1 from no and yes
final_dataframe["default"]=final_dataframe["default"].map({"no":0,"yes":1})
final_dataframe
#Train Test Splitting and normalizing the columns on the final dataframe
X=final_dataframe.drop("default",axis=1)
y=final_dataframe["default"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2,shuffle=True)
scaler=StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[('num', scaler, cont_cols)],remainder='passthrough')
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.fit_transform(X_test)
#Training a logistic regression model
model=LogisticRegression()
model.fit(X_train_scaled,y_train)
#Predicting the outcome on train and test data using Logistic Regression
y_pred_test=model.predict(X_test_scaled)
y_pred_train=model.predict(X_train_scaled)
print("Training data performance")
print(classification_report(y_train, y_pred_train))
print("Training Data Confusion Matrix")
print(confusion_matrix(y_train,y_pred_train))
print("Training Data Accuracy Score")
print(accuracy_score(y_train,y_pred_train))

print("Testing data performance")
print(classification_report(y_test, y_pred_test))
print("Testing Data Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))
print("Testing Data Accuracy Score")
print(accuracy_score(y_test,y_pred_test))
#Training a Quadratic Discriminant Analysis model 
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)
#Predicting the outcome on train and test data
y_pred_test=qda.predict(X_test_scaled)
y_pred_train=qda.predict(X_train_scaled)
print("Training data performance")
print(classification_report(y_train, y_pred_train))
print("Training Data Confusion Matrix")
print(confusion_matrix(y_train,y_pred_train))
print("Training Data Accuracy Score")
print(accuracy_score(y_train,y_pred_train))

print("Testing data performance")
print(classification_report(y_test, y_pred_test))
print("Testing Data Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))
print("Testing Data Accuracy Score")
print(accuracy_score(y_test,y_pred_test))
#Training a Support Vector Classifier model and predicting the outcome on train and test data
svm = SVC()
param_grid = {
    'C': [4, 10, 18, 30],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=skf, scoring='recall')
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_
y_pred_test=best_svm.predict(X_test_scaled)
y_pred_train=best_svm.predict(X_train_scaled)
print("Training data performance")
print(classification_report(y_train, y_pred_train))
print("Training Data Confusion Matrix")
print(confusion_matrix(y_train,y_pred_train))
print("Training Data Accuracy Score")
print(accuracy_score(y_train,y_pred_train)) 
print("Testing data performance")
print(classification_report(y_test, y_pred_test))
print("Testing Data Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))
print("Testing Data Accuracy Score")
print(accuracy_score(y_test,y_pred_test))
#Training a random forest classifier model and predicting the outcome on train and test data
rf_clf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300,400,500],
    'max_depth': [None, 10, 20,30],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 4,6]  
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_clf, param_grid, cv=skf, scoring='recall')
grid_search.fit(X_train_scaled, y_train)
best_rf_clf = grid_search.best_estimator_
y_pred_test=best_rf_clf.predict(X_test_scaled)
y_pred_train=best_rf_clf.predict(X_train_scaled)
print("Training data performance")
print(classification_report(y_train, y_pred_train))
print("Training Data Confusion Matrix")
print(confusion_matrix(y_train,y_pred_train))
print("Training Data Accuracy Score")
print(accuracy_score(y_train,y_pred_train))
print("Testing data performance")
print(classification_report(y_test, y_pred_test))
print("Testing Data Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))
print("Testing Data Accuracy Score")
print(accuracy_score(y_test,y_pred_test))
#Training a Gradient Boosting Classifier model and predicting the outcome on train and test data
gbc = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
grid_search = GridSearchCV(gbc, param_grid, cv=skf, scoring='recall')
grid_search.fit(X_train_scaled, y_train)
best_gbc = grid_search.best_estimator_
y_pred_test=best_gbc.predict(X_test_scaled)
y_pred_train=best_gbc.predict(X_train_scaled)
print("Training data performance")
print(classification_report(y_train, y_pred_train))
print("Training Data Confusion Matrix")
print(confusion_matrix(y_train,y_pred_train))
print("Training Data Accuracy Score")
print(accuracy_score(y_train,y_pred_train))
print("Testing data performance")
print(classification_report(y_test, y_pred_test))
print("Testing Data Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))
print("Testing Data Accuracy Score")
print(accuracy_score(y_test,y_pred_test))
#Displaying the Precision Recall Curve
classifiers=[model,qda,best_svm,best_rf_clf,best_gbc]
precision_list = []
recall_list = []
auc_pr_list = []
names=["Logistic Regression", "QDA","SVC","Random Forest","Gradient Boosting"]
for clf in classifiers:
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test_scaled)
    else:
        y_score = clf.predict_proba(X_test_scaled)[:, 1]
    if len(y_test) != len(y_score):
        y_test = y_test[:len(y_score)]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    auc_pr = auc(recall, precision)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_pr_list.append(auc_pr)
plt.figure(figsize=(8, 6))
for i, clf in enumerate(classifiers):
    plt.plot(recall_list[i], precision_list[i], label=f'{names[i]}(AUC-PR = {auc_pr_list[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
best_index = auc_pr_list.index(max(auc_pr_list))
best_model = classifiers[best_index]
print("Best model based on AUC-PR:", best_model)



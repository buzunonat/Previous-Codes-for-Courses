import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import datetime as dt
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,recall_score,precision_score,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder,TargetEncoder,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall

##### Functions
# this functio below is useful for encoding categoricals with NA values in the prediction
# does the slicing and gives the encoded columns without leaking
def category_encoding(colname, data):
    # create the encoder class
    encoder = TargetEncoder(target_type="continuous",
                            cv=10, smooth="auto", shuffle=False)
    # divide data into train and test depending on 'died_in_hospital' NA or not
    x_train_enc = data[colname][data.died_in_hospital.notna()].values.reshape(-1, 1)
    y_train_enc = data.died_in_hospital[data.died_in_hospital.notna()].values
    encoder.fit(x_train_enc, y_train_enc)

    # transform the column
    x_new = encoder.transform(data[colname].values.reshape(-1, 1))

    return x_new



## loading the data

host = "mysql-1.cda.hhs.se"
username = "7313"
password = "data"
schema = "Survivability"
connection_string = "mysql+pymysql://{}:{}@{}/{}".format(username, password, host, schema)
connection = create_engine(connection_string)
# Basic merging of two datasets.
sql_query1 = """
SELECT * 
FROM Patient
    LEFT JOIN Study 
        ON Patient.id = Study.patient_id
"""
# Since it was not possible to do any pivotting in MySQL other than manually using "CASE WHEN",
# i decided to import the whole dataset and do it here.
sql_query2 = """
SELECT *
FROM PatientExamination
"""


# load the datasets
df_1= pd.read_sql_query(con=connection.connect(), sql=(sql_query1))
df_2= pd.read_sql_query(con=connection.connect(), sql=(sql_query2))

# pivot the df_2 because the data is in a long format
df_2_wide = df_2.pivot(index="patient_id",columns="measurement",values="value").reset_index()
# since the values are turned into "object"s due to pivotting
# I want to change the numeric ones to float
categ_cols=["patient_id","Has Cancer","Zodiac Sign","Has Dementia","Has Diabetes"]
# turn every non category into floats
df_2_wide_float = df_2_wide.drop(columns=categ_cols).astype(float)
# turn categories into category
df_2_wide_char= df_2_wide[categ_cols].astype("category")
# merge
df_2_wide=pd.concat([df_2_wide_char,df_2_wide_float],axis=1)

# now that the formats are matching, lets merge the datasets into one
df=pd.merge(df_1,df_2_wide,how="left",left_on="id",right_on="patient_id")


# Variable formatting, editing dtypes
df["admission_date"] = pd.to_datetime(df["admission_date"]) # datetime conversion
df["study_entry_date"] = pd.to_datetime(df["study_entry_date"]) # datetime conversion
df["admission_month"] = df["admission_date"].dt.month # extracting month
df["admission_year"]= df["admission_date"].dt.year # extracting year
df["study_month"]= df["study_entry_date"].dt.month # extracting month
df["study_year"]= df["study_entry_date"].dt.year # extracting year
# changing datatypes into category
categories = ["gender","Has Dementia","Has Diabetes","disease_category","disease_class","income_bracket","admission_year"]
df[categories]=df[categories].astype("category")


# drop unnecessary columns
# patient_id_x etc happened because of the merging
df.drop(columns=["patient_id_x","patient_id_y",
                 "admission_date","study_entry_date"],inplace=True)


######## Question 1.1 ########

## ALL WHITE BLOOD CELLS DISTRIBUTION

sns.set(style="whitegrid")
plt.figure(figsize=(30, 15))
ax = sns.histplot(data=df, x="White Blood Cell Count",
                  kde=True,color="pink",edgecolor="black",bins=70)
ax.lines[0].set_color('blue')
mean_value = df["White Blood Cell Count"].mean()
median_value = df["White Blood Cell Count"].median()

plt.axvline(mean_value, color='purple', linestyle='--',
            linewidth=2, label=f'Mean: {mean_value:.2f}')

plt.axvline(median_value, color='darkgreen', linestyle='--',
            linewidth=2, label=f'Median: {median_value:.2f}')

plt.title("Distribution of White Blood Cell Count",
          fontsize=25, fontweight='bold')
plt.xlabel("White Blood Cell Count", fontsize=25)
#plt.xlim(-1, 70) if we want to see a better picture without outliers
plt.ylabel("Frequency", fontsize=25)
plt.legend(fontsize="30")
plt.show()


######## Question 1.2 ########

# first slice the ones we have to predict

df_pred = df[df["days_before_discharge"].isna()] # final prediction dataset

# create the file to be exported as JSON.
submission_df = pd.DataFrame({"patient_id":df_pred["id"],
                              "prediction_accuracy_model":None,
                              "white_blood_cell_count":df_pred["White Blood Cell Count"].round().astype("Int64"),
                              "prediction_profit_model":None
                              })



######## Question 2 ########

######## Question 2.1 ########



# Creating a dataset without NA's and with NA's
# no NA dataset will be used for ANN
# NA dataset will be used for XGboost

# the slicing is done using 'df_pred' because that dataset includes the final predictions,
# hence its NA values are more crutial
# NA's in the other slice can be handled.
df_NO_NA=df.loc[:,df_pred.isna().sum()==0]

# no NA removal for df_NA
df_NA=df


# since days_before_discharge were also dropped during the NA removal, I add it as 'class' set since
# its NA values holds the classification set
class_set = df["days_before_discharge"].copy(deep=True)


#Divide the dataset into two - categorical and numeric
categorical_cols = df_NO_NA.select_dtypes(include=['object', 'category']).columns
numerical_cols = df_NO_NA.select_dtypes(include=['int64', 'float64',"int32"]).columns
# Encode the categorical variables using the function defined previously
for col in categorical_cols:
    df_NO_NA.loc[:,col] = category_encoding(col,df_NO_NA)


# same process for the df_NA
categorical_cols2 = df_NA.select_dtypes(include=['object', 'category']).columns
numerical_cols2 = df_NA.select_dtypes(include=['int64', 'float64',"int32"]).columns
# Encode the categorical variables using the function defined previously
for col in categorical_cols2:
    df_NA.loc[:,col] = category_encoding(col,df_NA)




# variable selection for ANN
df_NO_NA.drop(columns = ["death_recorded_after_hospital_discharge",
                         "Zodiac Sign","days_in_hospital_before_study",
                         "study_month","study_year","id",
                         "existing_models_2_months_survival_prediction",
                         "existing_models_6_months_survival_prediction"
                         ],
              inplace=True)
# adding back the classification set
df_NO_NA["class_set"]=class_set

# dropping the similar columns for XGboost
df_NA.drop(columns = ["death_recorded_after_hospital_discharge",
                         "Zodiac Sign",
                         "days_in_hospital_before_study","study_month","study_year",
                         "id","language","days_of_follow_up","years_of_education"
                         ],
              inplace=True)




df_NO_NA_play = df_NO_NA[df_NO_NA["class_set"].notna()] # we are going to train with this one
df_NO_NA_pred = df_NO_NA[df_NO_NA["class_set"].isna()] # final prediction dataset
df_NO_NA_play=df_NO_NA_play.dropna() # there is a single row with an NA value, lets drop it

# same split for the df_NA
df_NA_play = df_NA[df_NA["days_before_discharge"].notna()]
df_NA_pred = df_NA[df_NA["days_before_discharge"].isna()]




X = df_NO_NA_play.drop(columns=["class_set", "died_in_hospital"]) # Setting up for training, independent variables
y = df_NO_NA_play["died_in_hospital"].values # dependent

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10) # split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.10) # split again for the validation set

scaler = StandardScaler() #scale the variables
x_train_preprocessed = scaler.fit_transform(x_train)  # fit using the training
x_valid_preprocessed = scaler.transform(x_valid) # transform so no leakage
x_test_preprocessed = scaler.transform(x_test) # transform so no leakage
# transform the classification set as well
x_real_preprocessed = scaler.transform(df_NO_NA_pred.drop(columns=["class_set", "died_in_hospital"]))


pca = PCA(n_components=0.90, svd_solver='full') # PCA with 90% explained variance
x_train_pca = pca.fit_transform(x_train_preprocessed) # fit transform
x_valid_pca = pca.transform(x_valid_preprocessed) # transform only
x_test_pca = pca.transform(x_test_preprocessed) # transform only
x_real_pca = pca.transform(x_real_preprocessed) # transform only

# ANN
model = Sequential()
# input shape should be equal to the num of cols
model.add(Input(shape=(x_train_pca.shape[1],)))
# 32 neurons, linear activation
model.add(Dense(32, activation='linear'))
# 64 neurons linear activation
model.add(Dense(64, activation='linear'))
# sigmoid output function since this is a classification problem
model.add(Dense(units=1, activation='sigmoid'))
#compiling the model
# loss is binary crossentropy
# adam optimizer
# additional metric : Recall
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=[Recall()])

# Early stopping to over fitting
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=20,
                               restore_best_weights=True)
model.fit(x=x_train_pca,
          y=y_train,
          epochs=200,
          batch_size=32,
          validation_data=(x_valid_pca, y_valid),
          callbacks=[early_stopping])
# Loss plot
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()
plt.show()
plt.savefig("ann_loss.pdf")






## training for XG boost

X_xg = df_NA_play.drop(columns=["days_before_discharge", "died_in_hospital"]) # Setting up for training, independent variables
y_xg = df_NA_play["died_in_hospital"].values # dependent

x_train_xg, x_test_xg, y_train_xg, y_test_xg = train_test_split(X_xg, y_xg, test_size=0.10) # split
# I retrained without valid datasets to get a better estimate hence they are commented out in the submission

#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.10) # split again for the validation set

scaler_xg = StandardScaler() #scale the variables
x_train_pre_xg = scaler.fit_transform(x_train_xg)  # fit using the training
#x_valid_preprocessed = scaler.transform(x_valid) # transform so no leakage
x_test_pre_xg = scaler.transform(x_test_xg)# transform so no leakage
# transform the classification set as well
x_real_pre_xg = scaler.transform(df_NA_pred.drop(columns=["days_before_discharge", "died_in_hospital"]))

# Maximizing hyperparameters
space = {
    #'n_estimators': hp.choice('n_estimators', range(100, 500)),
    'max_depth': hp.choice('max_depth', range(2, 30)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'min_child_weight': hp.choice('min_child_weight', range(2, 10)),
    'gamma': hp.uniform('gamma', 0, 20),
# I tried optimizing over other parameters but they did not contribute as much
    # and they were taking too much computing power given the limited time for the exam

    #'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
    #"subsample":hp.uniform('subsample', 0.45, 1),
    #"scale_pos_weight": hp.choice('scale_pos_weight', range(1, 80)),
    #"lambda":  hp.loguniform('lambda', np.log(0.01), np.log(0.2))
}
def objective(params):
    classifier = XGBClassifier(**params,n_estimators=252)
    #cross validation score is recall since it is crutial for the insurance estimation.
    recall = cross_val_score(classifier, x_train_pre_xg, y_train_xg,scoring="recall",
                               cv=5).mean()
    # Return the negative accuracy as we want to minimize the negative value
    return {'loss': -recall, 'status': 'ok'}



trials = Trials()
best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

# run the model using the optimized params
XGmodel =XGBClassifier(**best_params,n_estimators=250)

# fit
XGmodel.fit(x_train_pre_xg,y_train_xg,verbose=2)




######## Question 2.2 ########
ann_pred_3264=model.predict(x_test_pca)
threshold=0.5 # threshold probability for classification
## ANN predictions
recall_score(y_test,ann_pred_3264>threshold)
precision_score(y_test,ann_pred_3264>threshold)
accuracy_score(y_test,ann_pred_3264>threshold)
confusion_matrix(y_test,ann_pred_3264>threshold)

## XG predictions
pred_xg=XGmodel.predict_proba(x_test_pre_xg)[:,1]
accuracy_score(y_test_xg,pred_xg>threshold)
recall_score(y_test_xg,pred_xg>threshold)
precision_score(y_test_xg,pred_xg>threshold)
confusion_matrix(y_test_xg,pred_xg>threshold)



######## Question 2.3 ########
# final prediction of the classification set
predictions_ann = model.predict(x_real_pca).reshape(-1)
xg_predict_probs= XGmodel.predict_proba(x_real_pre_xg)[:,1]
# an average of the predictions
final_pred = (predictions_ann + xg_predict_probs)/2

# visualization of the prediction densities
plt.figure(figsize=(10, 6))

sns.kdeplot(predictions_ann, alpha=0.5, label='ANN Predictions')
sns.kdeplot(xg_predict_probs, alpha=0.5, label='XGBoost Probabilities')
sns.kdeplot(final_pred, alpha=0.5, label='Final Predictions')

plt.xlabel('Value')  # Probability
plt.ylabel('Density')
plt.title('Combined Density of the Predictions')
plt.legend()
plt.show()
plt.savefig("Combined.pdf")


#recording the predictions in the json dataframe
submission_df["prediction_accuracy_model"]=np.where(final_pred>0.5,1,0) # 0.5 is the threshold prob


######## Question 3 ########
######## Question 3.1 ########
cost = []
false_positives=[]
false_negatives = []
thresholds =np.arange(0.00, 1.0, 0.001).tolist()

# looping over different thresholds to estimate the cost of the insurance policy
# the strategy is to sell off every prediction of "dead". The misclassifications cause extra cost
# we calculate the threshold where its less costly
for thres in thresholds:
    true_positive = confusion_matrix(y_test, ann_pred_3264 > thres)[0, 0]
    false_positive = confusion_matrix(y_test, ann_pred_3264 > thres)[0, 1]
    false_negative = confusion_matrix(y_test, ann_pred_3264 > thres)[1, 0]
    true_negative = confusion_matrix(y_test, ann_pred_3264 > thres)[1, 1]
    cost_=false_positive * 150000 + true_negative * 150000 + false_negative * 500000
    cost.append(cost_)
    false_positives.append(false_positive)
    false_negatives.append(false_negative)


sns.relplot(x=thresholds, y=cost, kind="line")
plt.xlabel("Threshold Probabilities")
plt.ylabel("Cost of Policy")
plt.title("Cost of Policy vs. Threshold Probabilities")
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.grid(True)
plt.show()



#### Repeating the same thing for the XG boost prediction

pred_xg_prob=XGmodel.predict_proba(x_test_pre_xg)[:,1]
cost_xg = []
false_positives=[]
false_negatives = []
thresholds =np.arange(0.00, 1.0, 0.001).tolist()
for thres in thresholds:
    true_positive = confusion_matrix(y_test_xg, pred_xg_prob > thres)[0, 0]
    false_positive = confusion_matrix(y_test_xg, pred_xg_prob > thres)[0, 1]
    false_negative = confusion_matrix(y_test_xg, pred_xg_prob > thres)[1, 0]
    true_negative = confusion_matrix(y_test_xg, c > thres)[1, 1]
    cost_=false_positive * 150000 + true_negative * 150000 + false_negative * 500000
    cost_xg.append(cost_)
    false_positives.append(false_positive)
    false_negatives.append(false_negative)


sns.relplot(x=thresholds, y=cost, kind="line")
plt.xlabel("Threshold Probabilities")
plt.ylabel("Cost of Policy")
plt.title("Cost of Policy vs. Threshold Probabilities")
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.grid(True)
plt.show()








######## Question 3.2 ########

#the threshold is 0.35 to minimize the cost
submission_df["prediction_profit_model"]=np.where(final_pred>0.35,1,0)

######## Question 3.3 ########

# Cost of Doing Nothing
0.5 * y_test.sum() # 0.5 MEUR * num of dead

#Cost of Selling All Claims
0.15 * 1000 # 0.15 * 1000 ppl in the test set

#cost when profit maximizing

min(cost)+min(cost_xg)/2000000


######## Question 3.4 ########
# changing the threshold to see how errors change

# Setting up the subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plotting false negatives vs thresholds on the first subplot
sns.lineplot(x=thresholds, y=false_negatives, ax=axes[0])
axes[0].set_title('False Negatives vs Thresholds')
axes[0].set_xlabel('Thresholds')
axes[0].set_ylabel('False Negatives')
axes[0].grid(True)  # Adding grid

# Plotting false positives vs thresholds on the second subplot
sns.lineplot(x=thresholds, y=false_positives, ax=axes[1])
axes[1].set_title('False Positives vs Thresholds')
axes[1].set_xlabel('Thresholds')
axes[1].set_ylabel('False Positives')
axes[1].grid(True)

plt.tight_layout()
plt.show()


submission_df.to_json("42446_take_home_exam2.json",
                    orient="records", indent=2)
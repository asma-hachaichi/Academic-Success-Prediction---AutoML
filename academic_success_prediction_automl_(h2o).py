"""Academic Success Prediction - AutoML (H2O)"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Start the H2O cluster
h2o.init()

# Import train/test set into H2O
train = h2o.import_file("train.csv")
test = h2o.import_file("test.csv")

train.head()

test.head()

# Specify target and predictors
y = "Target"
x = train.columns
x.remove(y)

# Convert target column to factor
train[y] = train[y].asfactor()

# Run AutoML for 10 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)

#Predict target from test data
pred_h2o = aml.predict(test)

#Convert h2o data to dataframe
test_df = test.as_data_frame()
pred_df = pred_h2o.as_data_frame()

#Save prediction to csv file
pred = pd.DataFrame()
pred['id'] = test_df['id']
pred['Target'] = pred_df['predict']
pred.to_csv('/kaggle/working/submission.csv',index=False)
pred


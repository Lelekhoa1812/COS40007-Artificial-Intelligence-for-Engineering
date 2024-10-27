import joblib
import tensorflow as tf
import pandas as pd

# PERFORMANCE
model_per = tf.keras.models.load_model('best_ann_model_speed.h5')
# Display model summary (to get the input shape)
# model_per.summary()
X_train = pd.read_csv('data_performance.csv')  # Replace with your actual training data file path
feature_names = X_train.columns.tolist()
# Print the feature names used in the model
print("Features used in the model performance:", feature_names, "\n")

# SECTOR
model1 = joblib.load('model_top_sec.pkl')
if hasattr(model1, 'feature_names_in_'):
    print("Feature names in the model sector:", model1.feature_names_in_, "\n")
else:
    print("The model sector was not fitted with features.")

# SITE
model2 = joblib.load('model_top_site.pkl')
if hasattr(model2, 'feature_names_in_'):
    print("Feature names in the model site:", model2.feature_names_in_, "\n")
else:
    print("The model site was not fitted with features.")
import csv
import glob
from itertools import chain
from typing import Any, List

from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import matplotlib
import pandas as pd
import numpy as np
from IPython.core.display import display
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from util.config import load_config
from nnet import predict
from sklearn.model_selection import train_test_split
from dataset.pose_dataset import data_to_input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from util import visualize
import pickle

# import resize_images

cfg = load_config("pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read images from a path
pose_image_resources_downward = "../pose_images/Database_Resized/DownwardDog_mixed/*"
pose_image_resources_plank = "../pose_images/Database_Resized/Plank_mixed/*"
pose_image_resources_tree = "../pose_images/Database_Resized/Tree_mixed/*"
pose_image_resources_warrior = "../pose_images/Database_Resized/WarriorII_mixed/*"

pose_image_resources_all_right = "../pose_images/Database_Resized/all_poses_right/*"
# Uncomment this line and comment line before for development purposes (increase time execution)£
#
# pose_image_resources = "../pose_images/acc/*.jpeg"  # 26 samples 6 testing set --> Score 0,767 (n_estimators=40, max_depth=20) 0,916
# pose_image_resources ="../pose_images/all_tree/*.jpeg"
# Images normalization --> using resize_images.py script

features = []
picture_name = []


# Read all images, call cnn model and make predictions about human main body parts
for images in glob.glob(pose_image_resources_warrior):
    try:
        image_name = images.title()
        image = plt.imread(images)
        picture_name.append(image_name)
        image_batch: ndarray = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose: ndarray = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        # print(pose.toarr)

        # Visualise
        # visualize.show_heatmaps(cfg, image, scmap, pose)
        # visualize.waitforbuttonpress()

        features_df = list(chain.from_iterable(pose))
        y0 = '_'
        features_df.append(y0)
        y1 = '_'
        features_df.append(y1)
        features.append(features_df)

        # print(features_df)
        # this needs reshape --> features_pandas = pd.DataFrame(features_df, columns=labels)
        # dataFrame from 1d nparray --> df = pd.DataFrame(a.reshape(-1, len(a)), columns=labels)
    except Exception as e:
        print(e)

labels: List[Any] = ['x_ankle_l', 'y_ankle_l', 'error_ankle_l',
                     'x_ankle_r', 'y_ankle_r', 'error_ankle_r',
                     'x_knee_l', 'y_knee_l', 'error_knee_l',
                     'x_knee_r', 'y_knee_r', 'error_knee_r',
                     'x_hip_l', 'y_hip_l', 'error_hip_l',
                     'x_hip_r', 'y_hip_r', 'error_hip_r',
                     'x_wrist_l', 'y_wrist_l', 'error_wrist_l',
                     'x_wrist_r', 'y_wrist_r', 'error_wrist_r',
                     'x_elbow_l', 'y_elbow_l', 'error_elbow_l',
                     'x_elbow_r', 'y_elbow_r', 'error_elbow_r',
                     'x_shoulder_l', 'y_shoulder_l', 'error_shoulder_l',
                     'x_shoulder_r', 'y_shoulder_r', 'error_shoulder_r',
                     'x_chin', 'y_chin', 'error_chin',
                     'x_forehead', 'y_forehead', 'error_forehead', 'y0', 'y1']  # 44

features = np.asarray(features)
features_df: DataFrame = pd.DataFrame(features)
features_df.columns = labels

features_df['picture_name'] = picture_name

boolean_dict = {'Right': 1, 'Wrong': 0}
features_df.loc[features_df['picture_name'].str.contains('Right'), 'is_right'] = 1
features_df.loc[features_df['picture_name'].str.contains('Wrong'), 'is_right'] = 0
# features_df['is_right_again_2'] = features_df['picture_name'].apply(lambda x: 1 if features_df['picture_name']\
# .str.contains('Right') else 0)

poses_dict = {'Downward': 1, 'Plank': 2, 'Tree': 3, 'Warrior': 4}
features_df.loc[features_df['picture_name'].str.contains('Downward'), 'pose'] = 1
features_df.loc[features_df['picture_name'].str.contains('Plank'), 'pose'] = 2
features_df.loc[features_df['picture_name'].str.contains('Tree'), 'pose'] = 3
features_df.loc[features_df['picture_name'].str.contains('Warrior'), 'pose'] = 4
features_df.loc[~features_df['picture_name'].str.contains(r'Downward|Plank|Tree|Warrior'), 'pose'] = 5

# features_df.to_csv('prepared_data_499.csv')

# Create a column from the list
# features_df['y1'] = y1
# features_df = features_df.append(y1)

pd.set_option('display.max_columns', None)
# display(features_df.head(10))


# Guardar las coordenadas y pasarselo al RF o al SVM
# TODO: guardar los 5 resultados indep
features_df.to_csv('prepared_data_all_right.csv')

# Load data
# data = pd.read_csv(file_name, sep=';')
#
# # Display records
# display(data.head(n=2))
#
# data.isnull().any()
#
# data.describe()
#
# data.info()
#
# # More additional data analysis
# display(np.round(data.describe()))

# display(features_df.isnull().any())
# display(pd.plotting.scatter_matrix(features_pd_df, figsize=(40, 40), diagonal='kde'))
# plt.show()
# display(features_pd_df.head(n=3))
# display(features_pd_df.describe(include="all"))


# Split the data into features and target label
# features_df = features_df.drop(['picture_name', 'y0', 'y1', 'x_knee_l', 'y_knee_l', 'error_knee_l', 'x_knee_r', 'y_knee_r', 'error_knee_r'], axis=1)
features_df = features_df.drop(['picture_name', 'y0', 'y1'], axis=1)

# FEATURES PARA MODEL 1
pose_raw = features_df['pose']
features_without_pose_raw = features_df.drop(['pose'], axis=1)

# FEATURES FOR MODEL 2
is_right_raw = features_df['is_right']
features_without_is_right_raw = features_df.drop(['is_right'], axis=1)
# tree pose right and wrong --> Score 0.4148717948717949 , 258 samples


# Split the 'features' and 'income' data into training and testing sets FOR MODEL 1
# X_train, X_test, y_train, y_test = train_test_split(features_without_pose_raw,
                                                    #pose_raw,
                                                    #test_size=0.33,
                                                    #random_state=42)

# Split the 'features' and 'income' data into training and testing sets FOR MODEL 2
X_train, X_test, y_train, y_test = train_test_split(features_without_is_right_raw,
                                                    is_right_raw,
                                                    test_size=0.33,
                                                    random_state=42)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Initialize the randomForest model
# TODO: research how the model works with and without error variables
# Import any three supervised learning classification models from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Initialize the models
random_forest_class_model = RandomForestClassifier(n_estimators=40, max_depth=20)
random_forest_class_model.fit(X_train, y_train)

logistic_regression_model = LogisticRegression(random_state=42, solver='sag', multi_class='multinomial', max_iter=10000)
logistic_regression_model.fit(X_train, y_train)

# Save the model


# pickle.dumps(logistic_regression_model, open("saved_rfc_model.p", "wb"))

y_pred_rf = random_forest_class_model.predict(X_test)
y_pred_lr = logistic_regression_model.predict(X_test)


sco = random_forest_class_model.score(X_test, y_test)
R2 = r2_score(y_test, y_pred_rf)
print("R2 Random Forest: ", R2)
print("Score Random forest: ", sco)
Lr_R2 = r2_score(y_test, y_pred_lr)
print("R2 Logistic Reg: ", Lr_R2)

confusion_matrix(y_test, y_pred_lr)

cm_rf = confusion_matrix(y_test, y_pred_rf)
display(cm_rf)

acc_test = accuracy_score(y_test, y_pred_rf)

display(acc_test)

import seaborn as sn

plt.figure(figsize=(10, 10))
plt.title('Warrior: Confusion matrix - Random Forest Classifier')
plt.xticks(ha='center', va='top')
# MODEL 1
# sn.heatmap(cm_rf, cmap='Pastel1', xticklabels=poses_dict, yticklabels=poses_dict, annot=True, cbar=False)

# MODEL 2
sn.heatmap(cm_rf, cmap='Pastel1', xticklabels=boolean_dict, yticklabels=boolean_dict, annot=True, cbar=False)

plt.show()

# CALCULATING AUC ROC CURVE
from collections import Counter
display(Counter(y_test))

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_rf, pos_label=0)

# Print ROC curve
plt.plot(fpr, tpr)
plt.show()

# Print AUC
auc = np.trapz(tpr, fpr)
print('AUC:', auc)


def write_csv(position_data, file_name, save_error=True):
    if save_error:
        out = csv.writer(open(file_name, "a"), delimiter=',')
        out.writerow(position_data)
    else:
        pass
    # Pose, ankle, knee, hip, wrist, elbow, shoulder, chin, forehead --> 42 coord

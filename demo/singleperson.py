import csv
import glob
from itertools import chain
from typing import Any, List

import matplotlib
import pandas as pd
import numpy as np
from IPython.core.display import display
from numpy.core._multiarray_umath import ndarray
from pandas import DataFrame

from pyspark.mllib.tree import RandomForest

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
#import resize_images

cfg = load_config("pose_cfg.yaml")
file_name = "position_data_resized_witherrors_2.csv"


# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read images from a path
pose_image_resources = "../pose_images/DownwardDog/*.jpeg"

# Images normalization --> using resize_images.py script

features = []
indices = []

# Read all images, call cnn model and make predictions about human main body parts
for images in glob.glob(pose_image_resources):
    try:
        image_name = images.title()
        image = plt.imread(images)
        indices.append(image_name)
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

        # print(features_df)
        features.append(features_df)
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
                             'x_forehead', 'y_forehead', 'error_forehead', 'y0', 'y1']

features = np.asarray(features)
features_df: DataFrame = pd.DataFrame(features)
features_df.columns = labels

features_df['index'] = indices
features_df = features_df.set_index('index')

#features_df = features_df.
    #.assign(Pose=features_df['Index'])
              #+ ['Pose'] + ['Status']
display(features_df.head(10))
features_df.to_csv('prepared_data.csv')

#features_pd_df = pd.read_csv('position_data_resized_witherrors_2.csv', sep=",", index_col=1)
pd.set_option('display.max_columns', None)
#display(features_pd_df.isnull().any())
# display(pd.plotting.scatter_matrix(features_pd_df, figsize=(40, 40), diagonal='kde'))
# plt.show()
# display(features_pd_df.head(n=3))
# display(features_pd_df.describe(include="all"))
# Guardar las coordenadas y pasarselo al RF o al SVM


from sklearn.model_selection import train_test_split

# Split the data into features and target label
data_raw = features_df
features_raw = features_df.drop(['index'], axis=1)
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
                                                    data_raw,
                                                    test_size=0.1,
                                                    random_state=42)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# # Load dataset
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
# # Some more additional data analysis
# display(np.round(data.describe()))

# Initialize the randomForest model
# TODO: ver como se comporta el modelo con error y sin error
# X_train=
# y_train=
# reg = RandomForestRegressor(min_samples_leaf=9, n_estimators=100)
# reg.fit(X_train, y_train)


def write_csv(position_data, file_name, save_error=True):
    if save_error:
        out = csv.writer(open(file_name, "a"), delimiter=',')
        out.writerow(position_data)
    else:
        pass
    # Pose, ankle, knee, hip, wrist, elbow, shoulder, chin, forehead --> 42 coord

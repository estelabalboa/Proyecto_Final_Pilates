import csv
import glob
from itertools import chain
from typing import Any, List

import matplotlib
import pandas as pd
from IPython.core.display import display
from numpy.core._multiarray_umath import ndarray

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

cfg = load_config("pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read images from a path
pose_image_resources = "../pose_images/resized/*"


def write_csv(position_data, file_name, save_error=True):
    if save_error:
        out = csv.writer(open(file_name, "a"), delimiter=',', quoting=csv.QUOTE_ALL)
        out.writerow(position_data)
    else:
        pass
    # Pose, ankle, knee, hip, wrist, elbow, shoulder, chin, forehead --> 42 coordenadas


for images in glob.glob(pose_image_resources):
    try:
        image = plt.imread(images)

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})

        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose: ndarray = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        # Visualise
        # visualize.show_heatmaps(cfg, image, scmap, pose)
        # visualize.waitforbuttonpress()

        file_name = "position_data_resized_witherrors.csv"
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
                             'x_forehead', 'y_forehead', 'error_forehead']

        features_df: List[Any] = (list(chain.from_iterable(pose)))
        print(labels)
        print(features_df)
        # write_csv(features_df, file_name)
        # TODO: provisional, queremos guardar el pandasDF como CSV


        # features_pd_df.toCSV(file_name, sep=',')

        # TODO: normalizar las imagenes, entre las dimensiones de las imágenes--> hacemos el script resize_images.py

    except Exception as e:
        print(e)

features_pd_df = pd.read_csv('position_data_resized_witherrors.csv', sep=",")
display(features_pd_df.describe(include="all"))
## Guardar las coordenadas y pasarselo al RF o al SVM

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
# display(pd.np.round(data.describe()))

# Initialize the randomForest model
# TODO: ver como se comporta el modelo con error y sin error
# X_train=
# y_train=
# reg = RandomForestRegressor(min_samples_leaf=9, n_estimators=100)
# reg.fit(X_train, y_train)

import csv
import glob
from itertools import chain
from typing import Any, List

import matplotlib
import pandas as pd
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

# Read image from file
# file_name = "image.png"
# image = plt.imread(file_name, format='RGB')

# Read images from a path
pose_image_resources = "../pose_images/*jpeg"


def write_csv(position_data, save_error=True):
    if save_error == True:
        out = csv.writer(open("position_data_errors.csv", "a"), delimiter=',', quoting=csv.QUOTE_ALL)
        out.writerow(position_data)
    else:
        pass
    # Pose, ankle, knee, hip, wrist, elbow, shoulder, chin, forehead --> 42 coordenadas


for images in glob.glob(pose_image_resources):
    try:
        image = plt.imread(images, format="RGB")

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose: ndarray = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        features_df: List[Any] = list(chain.from_iterable(pose))
        write_csv(features_df)
        # TODO: provisional, queremos guardar el pandasDF como CSV

        labels = ['x_ankle_l', 'y_ankle_l', 'error_ankle_l',
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

        features_pd_df = pd.DataFrame(features_df, columns=labels)

        features_pd_df.toCSV(r'position_data_witherrors.csv', sep=',')

        # print(features_df)

        # TODO: normalizar las imagenes, entre las dimensiones de las imágenes

        # Visualise
        visualize.show_heatmaps(cfg, image, scmap, pose)
        # visualize.waitforbuttonpress()
    except Exception as e:
        print(e)

## Guardar las coordenadas y pasarselo al RF o al SVM

# Initialize the randomForest model
# TODO: ver como se comporta el modelo con error y sin error
# X_train=
# y_train=
# reg = RandomForestRegressor(min_samples_leaf=9, n_estimators=100)
# reg.fit(X_train, y_train)

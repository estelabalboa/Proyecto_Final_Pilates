import glob

from matplotlib.pyplot import *
import matplotlib

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
#file_name = "image.png"
#image = plt.imread(file_name, format='RGB')

# Read images from a path
pose_image_resources = "../pose_images/*.jpeg"


for images in glob.glob(pose_image_resources):
    try:
        image = plt.imread(images, format="RGB")

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        # Visualise
        visualize.show_heatmaps(cfg, image, scmap, pose)
        visualize.waitforbuttonpress()
    except Exception as e:
        print(e)



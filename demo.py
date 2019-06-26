import keras.backend as K
import cv2
from networks.faceswap_gan_model import FaceswapGANModel

import numpy as np
K.set_learning_phase(0)
RESOLUTION = 128 # 64x64, 128x128, 256x256
assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, 256"

# Architecture configuration
arch_config = {}
arch_config['IMAGE_SHAPE'] = (RESOLUTION, RESOLUTION, 3)
arch_config['use_self_attn'] = True
arch_config['norm'] = "instancenorm" # instancenorm, batchnorm, layernorm, groupnorm, none
arch_config['model_capacity'] = "standard" # standard, lite

model = FaceswapGANModel(**arch_config)

model.load_weights(path="./endeweight")

from converter.video_converter  import VideoConverter

from detector.face_detector import MTCNNFaceDetector
mtcnn_weights_dir = "./mtcnn_weights/"
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

# from detector.s3fd_face_detector import S3FDFaceDetector
# fd = S3FDFaceDetector(K.get_session(), mtcnn_weights_dir)

# from detector.small_face_detector import SmallFaceDetector
# weights_dir = "./models/pickle_model/test.pickle"
# fd = SmallFaceDetector(K.get_session(), weights_dir)
frames = 0
x0 = x1 = y0 = y1 = 0
vc = VideoConverter(x0,x1,y0,y1,frames)

vc.set_face_detector(fd)
vc.set_gan_model(model)

options = {
    # ===== Fixed =====
    "use_smoothed_bbox": True,
    "use_kalman_filter": True,
    "use_auto_downscaling": True,
    "bbox_moving_avg_coef": 0.60,
    "min_face_area": 40 * 40,
    "IMAGE_SHAPE": model.IMAGE_SHAPE,
    # ===== Tunable =====
    "kf_noise_coef": 5e-2,
    "use_color_correction": "adain_xyz",#"hist_match",#adain_xyz
    "detec_threshold": 0.80,#60
    "roi_coverage": 0.95,
    "enhance": 0.,
    "output_type": 1,
    "direction": "BtoA",
}

duration = None
input_fn = "star-wars0.mp4"
output_fn = "test.mp4"

vc.convert(input_fn=input_fn, output_fn=output_fn, options=options, duration=duration)



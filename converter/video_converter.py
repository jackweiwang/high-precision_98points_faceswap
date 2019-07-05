from .kalman_filter import KalmanFilter
from .landmarks_alignment import *
from .face_transformer import FaceTransformer
from .vc_utils import *
import numpy as np
from moviepy.editor import VideoFileClip
import math
import face98
#import torch
class VideoConverter(object):
    """
    This class is for video conversion

    Attributes:
        ftrans: FaceTransformer instance
        fdetect: MTCNNFaceDetector instance
        prev_x0, prev_x1, prev_y0, prev_y1, frames: Variables for smoothing bounding box
        kf0, kf1: KalmanFilter instances for smoothing bounding box
    """

    def __init__(self, x0, x1, y0, y1,frames):
        # Variables for smoothing bounding box
        self.prev_x0 = x0
        self.prev_x1 = x1
        self.prev_y0 = y0
        self.prev_y1 = y1
        self.frames = frames

        # face transformer
        self.ftrans = FaceTransformer()

        # MTCNN face detector
        self.fdetect = None

        # Kalman filters
        self.kf0 = None
        self.kf1 = None
        self.fa = face98.Get_alignment()
        #self.sp = dlib.shape_predictor('./detector/models/shape_predictor_68_face_landmarks.dat')
    def set_gan_model(self, model):
        self.ftrans.set_model(model)

    def set_face_detector(self, fdetect):
        self.fdetect = fdetect

    def _get_smoothed_coord(self, x0, x1, y0, y1, img_shape, use_kalman_filter=True, ratio=0.65):
        if not use_kalman_filter:
            x0 = int(ratio * self.prev_x0 + (1 - ratio) * x0)
            x1 = int(ratio * self.prev_x1 + (1 - ratio) * x1)
            y1 = int(ratio * self.prev_y1 + (1 - ratio) * y1)
            y0 = int(ratio * self.prev_y0 + (1 - ratio) * y0)
        else:
            x0y0 = np.array([x0, y0]).astype(np.float32)
            x1y1 = np.array([x1, y1]).astype(np.float32)
            self.kf0.correct(x0y0)
            pred_x0y0 = self.kf0.predict()
            self.kf1.correct(x1y1)
            pred_x1y1 = self.kf1.predict()
            x0 = np.max([0, pred_x0y0[0][0]]).astype(np.int)
            x1 = np.min([img_shape[0], pred_x1y1[0][0]]).astype(np.int)
            y0 = np.max([0, pred_x0y0[1][0]]).astype(np.int)
            y1 = np.min([img_shape[1], pred_x1y1[1][0]]).astype(np.int)
            if x0 == x1 or y0 == y1:
                x0, y0, x1, y1 = self.prev_x0, self.prev_y0, self.prev_x1, self.prev_y1
        return x0, x1, y0, y1

    def _set_prev_coord(self, x0, x1, y0, y1):
        self.prev_x0 = x0
        self.prev_x1 = x1
        self.prev_y1 = y1
        self.prev_y0 = y0

    def _init_kalman_filters(self, noise_coef):
        self.kf0 = KalmanFilter(noise_coef=noise_coef)
        self.kf1 = KalmanFilter(noise_coef=noise_coef)

    def convert(self, input_fn, output_fn, options, duration=None):
        self.check_options(options)

        if options['use_kalman_filter']:
            self._init_kalman_filters(options["kf_noise_coef"])

        self.frames = 0
        self.prev_x0 = self.prev_x1 = self.prev_y0 = self.prev_y1 = 0

        if self.fdetect is None:
            raise Exception(f"face detector has not been set through VideoConverter.set_face_detector() yet.")

        clip1 = VideoFileClip(input_fn)
        if type(duration) is tuple:
            clip = clip1.fl_image(lambda img: self.process_video(img, options)).subclip(duration[0], duration[1])
        else:
            clip = clip1.fl_image(lambda img: self.process_video(img, options))
        clip.write_videofile(output_fn, audio=True)
        clip1.reader.close()
        try:
            clip1.audio.reader.close_proc()
        except:
            pass

    def process_video(self, input_img, options):
        """Transform detected faces in single input frame."""
        image = input_img

        # detect face using MTCNN (faces: face bbox coord, pnts: landmarks coord.)
        faces = self.fdetect.detect_face(image, minsize=40,
                                               threshold=options["detec_threshold"],
                                               factor=0.709,
                                               use_auto_downscaling=options["use_auto_downscaling"],
                                               min_face_area=options["min_face_area"]
                                               )

        # check if any face detected
        if len(faces) == 0:
            comb_img = get_init_comb_img(input_img)
            triple_img = get_init_triple_img(input_img, no_face=True)

        # init. output image
        mask_map = get_init_mask_map(image)
        comb_img = get_init_comb_img(input_img)
        change_num = 20
        best_conf_score = 0
        ratio_num = 0
        for i, (x0, y1, x1, y0, conf_score) in enumerate(faces):
            #lms = pnts[:, i:i + 1]
            x0 = x0 - change_num
            y0 = y0 - change_num
            x1 = x1 + change_num
            y1 = y1 + change_num

            # smoothe the bounding box
            if options["use_smoothed_bbox"]:
                if self.frames != 0 and conf_score >= best_conf_score:
                    x0, x1, y0, y1 = self._get_smoothed_coord(
                        x0, x1, y0, y1,
                        img_shape=image.shape,
                        use_kalman_filter=options["use_kalman_filter"],
                        ratio=options["bbox_moving_avg_coef"],
                    )
                    self._set_prev_coord(x0, x1, y0, y1)
                    best_conf_score = conf_score
                    self.frames += 1
                elif conf_score <= best_conf_score:
                    self.frames += 1
                else:
                    if conf_score >= best_conf_score:
                        self._set_prev_coord(x0, x1, y0, y1)
                        best_conf_score = conf_score
                    if options["use_kalman_filter"]:
                        for i in range(200):
                            self.kf0.predict()
                            self.kf1.predict()
                    self.frames += 1

            try:
                det_mask1 = input_img[int(x0):int(x1), int(y0):int(y1), :]
                srcheight = det_mask1.shape[0]
                srcwidth = det_mask1.shape[1]
                ratioh = srcheight/256
                ratiow = srcwidth/256
                #det_mask1 = cv2.imread('average3.png')
                det_mask = cv2.resize(det_mask1, (256, 256))

                n = self.fa.alignment(det_mask)
                preds = n.reshape((98, 2))
                #print(",".join(str(i) for i in src_list))

                centerPoint = [preds[54][1]*ratioh, preds[54][0]*ratiow ]
                # x, y, w, h = cv2.boundingRect(masksrc)
                # rec = np.abs(w-h)/2
                y0 = y0 + centerPoint[1] - srcwidth/2
                x0 = x0 + centerPoint[0] - srcheight/2
                y1 = y0 + srcwidth
                x1 = x0 + srcheight
                print(x0, x1, y0, y1)
                if(x0 < 0):
                    x0 = 0
                if (y0 < 0):
                    y0 = 0
                det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]
                #cv2.imwrite('det_face_im.png', det_face_im)
                srcheight = det_face_im.shape[0]
                srcwidth = det_face_im.shape[1]
                det_face_im1 = cv2.resize(det_face_im, (256, 256))
                n1 = self.fa.alignment(det_face_im1)
                preds = n1.reshape((98, 2))
                src_list = preds.tolist()
                if preds is not None:
                    # preds = preds[0]
                    mask = np.zeros_like(det_face_im1)

                    pnts_right = [(preds[i]) for i in range(0, 33)]
                    pnts_right1 = [preds[i] for i in range(33, 37)]
                    pnts_right2 = [preds[i] for i in range(42, 47)]

                    pnts_num = np.array(pnts_right)
                    pnts_num1 = np.array(pnts_right1)
                    pnts_num2 = np.array(pnts_right2)

                    pnts_num1[:, 1] = pnts_num1[:, 1] - 20
                    # pnts_num1[:,0] = pnts_num1[:,0] - 10
                    pnts_num2[:, 1] = pnts_num2[:, 1] - 20
                    # pnts_num2[:,0] = pnts_num2[:,0] + 10
                    pnt = np.vstack((pnts_num, pnts_num1, pnts_num2))

                    hull = cv2.convexHull(pnt).astype(np.int32)
                    mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                    mask = cv2.resize(mask, (srcwidth, srcheight))

                RotateMatrix = warp_im(src_list)

                # face transform
                r_im, r_rgb, r_a = self.ftrans.transform(
                    det_face_im,
                    direction=options["direction"],
                    roi_coverage=options["roi_coverage"],
                    color_correction=options["use_color_correction"],
                    IMAGE_SHAPE=options["IMAGE_SHAPE"]
                )

                rev_aligned_det_face_im = alignlandmarks(r_im, RotateMatrix)
                rev_aligned_det_face_im_rgb = alignlandmarks(r_rgb, RotateMatrix)
                rev_aligned_mask = alignlandmarks(r_a, RotateMatrix)

                # merge source face and transformed face
                result = np.zeros_like(det_face_im)
                #result = rev_aligned_det_face_im
                result = rev_aligned_mask / 255 * rev_aligned_det_face_im_rgb + (
                            1 - rev_aligned_mask / 255) * rev_aligned_det_face_im

                result = dealignlandmarks(result, RotateMatrix)

                resultint = result.astype(np.uint8) & mask
                result2 = det_face_im.astype(np.uint8)#  & (255-masksrc)
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                x, y, w, h = cv2.boundingRect(mask)

                
                center = (int(x + w / 2), int(y + h / 2))
                result = cv2.seamlessClone(resultint, result2, mask, center, cv2.NORMAL_CLONE )

                result_a = rev_aligned_mask
            except:
                # catch exceptions for landmarks alignment errors (if any)
                print(f"Face alignment error occurs at frame {self.frames}.")
                # get detected face
                det_face_im = input_img[int(x0):int(x1), int(y0):int(y1), :]

                result, _, result_a = self.ftrans.transform(
                    det_face_im,
                    direction=options["direction"],
                    roi_coverage=options["roi_coverage"],
                    color_correction=options["use_color_correction"],
                    IMAGE_SHAPE=options["IMAGE_SHAPE"]
                )

            comb_img[int(x0):int(x1), input_img.shape[1] + int(y0):input_img.shape[1] + int(y1), :] = result

            # Enhance output
            if options["enhance"] != 0:
                comb_img = -1 * options["enhance"] * get_init_comb_img(input_img) + (1 + options["enhance"]) * comb_img
                comb_img = np.clip(comb_img, 0, 255)

            if conf_score >= best_conf_score:
                mask_map[int(x0):int(x1), int(y0):int(y1), :] = result_a
                mask_map = np.clip(mask_map + .15 * input_img, 0, 255)
                # Possible bug: when small faces are detected before the most confident face,
                #               the mask_map will show brighter input_img
            else:
                mask_map[int(x0):int(x1), int(y0):int(y1), :] += result_a
                mask_map = np.clip(mask_map, 0, 255)

            triple_img = get_init_triple_img(input_img)
            triple_img[:, :input_img.shape[1] * 2, :] = comb_img
            #triple_img[:, input_img.shape[1] * 2:, :] = mask_map

        if options["output_type"] == 1:
            #cv2.imwrite('test.jpg', comb_img[:, input_img.shape[1]:, :])
            return comb_img[:, input_img.shape[1]:, :]  # return only result image
        elif options["output_type"] == 2:
            return comb_img  # return input and result image combined as one
        elif options["output_type"] == 3:
            return triple_img  # return input,result and mask heatmap image combined as one

    @staticmethod
    def check_options(options):
        if options["roi_coverage"] <= 0 or options["roi_coverage"] >= 1:
            raise ValueError(f"roi_coverage should be between 0 and 1 (exclusive).")
        if options["bbox_moving_avg_coef"] < 0 or options["bbox_moving_avg_coef"] > 1:
            raise ValueError(f"bbox_moving_avg_coef should be between 0 and 1 (inclusive).")
        if options["detec_threshold"] < 0 or options["detec_threshold"] > 1:
            raise ValueError(f"detec_threshold should be between 0 and 1 (inclusive).")
        if options["use_smoothed_bbox"] not in [True, False]:
            raise ValueError(f"use_smoothed_bbox should be a boolean.")
        if options["use_kalman_filter"] not in [True, False]:
            raise ValueError(f"use_kalman_filter should be a boolean.")
        if options["use_auto_downscaling"] not in [True, False]:
            raise ValueError(f"use_auto_downscaling should be a boolean.")
        if options["output_type"] not in range(1, 4):
            ot = options["output_type"]
            raise ValueError(f"Received an unknown output_type option: {ot}.")

import os
import cv2
from models import *
from utils import dataload,train_eval_utils
import torch
import torch.nn.functional as F

use_dataset = 'WFLW'
use_epoch = '900'
# load network
#devices = torch.device('cuda:0')
devices = 'cpu'
print('*****  WFLW trained Model Evaluating  *****')
print('Loading network ...')

class Get_alignment():

    def __init__(self):


        self.estimator = Estimator()
        self.regressor = Regressor(output=2 * 98)
        self.estimator = train_eval_utils.load_weights(self.estimator, 'detector/models/estimator_'+use_epoch+'.pth', devices)
        self.regressor = train_eval_utils.load_weights(self.regressor, 'detector/models/' + use_dataset+'_regressor_'+use_epoch+'.pth', devices)
        #self.estimator = self.estimator.cuda(device=devices)
        #self.regressor = self.regressor.cuda(device=devices)
        self.estimator.eval()
        self.regressor.eval()

    def alignment(self,face_img):

        face_gray = dataload.convert_img_to_gray(face_img)
        face_norm = dataload.pic_normalize(face_gray)

        input_face = torch.Tensor(face_norm)
        input_face = input_face.unsqueeze(0)
        input_face = input_face.unsqueeze(0).cpu()

        pred_heatmaps = self.estimator(input_face)
        pred_coords = self.regressor(input_face, pred_heatmaps[-1].detach()).detach().cpu().squeeze().numpy()


        return pred_coords

if __name__ == '__main__':
    ali = Get_alignment()

    face_img = cv2.imread('1.png')
    #print(face_img.shape)

    point = ali.alignment(face_img)
    torch.cuda.empty_cache()
    print(point)

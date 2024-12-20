from scipy.spatial.distance import cosine,euclidean, cityblock, minkowski, correlation, jaccard
import cv2
import numpy as np
import time
import gym
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from gym import spaces
from REID.Extractor import Extractor
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from detector import detect
import UdpComms as U
import warnings
from collections import deque
import collections
from PIL import Image
import pickle
import sys
import numpy as np
import math
import tqdm
import argparse
import glob
import os
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager
from fastreid.predictor import FeatureExtractionDemo
sys.path.append("..")
warnings.filterwarnings("ignore")

sock = U.UdpComms(udpIP="127.0.0.1", portTX=8011, portRX=8012, enableRX=True, suppressWarnings=True)
bbox_deque = collections.deque(maxlen=4)

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class fakeCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# 模拟命令行参数
fake_args_dict = {
    "config_file": "fastreid/configs/VeRi/VeRi_sbs_R50-ibn.yml",
    "parallel": False,
    "input": [],
    "output": "reid_output_folder",
    "opts": []
}
# 创建Namespace对象来模拟命令行参数
args = fakeCfg(**fake_args_dict)

class UnityCameraControlEnv(gym.Env):

    def __init__(self, testing=False, interactive=False, a_p=0.15, a_r=0.1,
                 e_thres=0.6):
        self.a_r = a_r
        self.a_p = a_p
        self.e_thres = e_thres

        # Size of the actual input to the CNN
        self.observation_size_h = 540  # 1920
        self.observation_size_w = 960  # 1080
        self.obs_vector = None
        self.current_center = [self.observation_size_w / 2, self.observation_size_h / 2]

        self.observation = np.zeros((self.observation_size_h, self.observation_size_w, 3,))
        self.detection_result = np.zeros((self.observation_size_h, self.observation_size_w, 3,))
        self.observation_space = spaces.Discrete(8) # 10
        low_angle = np.array([0.3], dtype=np.float32)
        high_angle = np.array([5], dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(7),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32),
                                          spaces.Box(low=low_angle, high=high_angle, dtype=np.float32)))
        self.error_memory = []
        self.epoch_counter = -1
        self.confidence_list = []
        self.action_time_list = []
        self.action_list = []
        self.target_area_list = []
        self.cam_fov_list = []
        self.iteration_counter = 0
        self.end_iteration = 0
        self.current_state_text = ''
        self.forward_action = 0
        self.forward_action_time = 0
        self.forward_x = 0
        self.forward_y = 0
        self.forward_a = 0
        self.forward_c = 0
        self.forward_pos_punish = 0
        self.forward_area_punish = 0
        self.forward_con_punish = 0
        self.forward_inbox_punish = 0
        self.action_text = 'None'
        self.action_time_text = 'None'
        self.interactive = interactive
        self.score_thr = 0.3
        self.object_num = -1
        self.rtn_object_num = -1
        self.miss_target_num = 0
        self.init_area_ratio = 0
        self.object_speed = -1
        self.envSeed = -2
        self.object_detect_num = 0
        self.discount_factor = 1.01
        self.THRESHOLD = 0.525
        self.in_bbox = False
        self.action_log_path = './action_log.txt'
        self.file_path = '/home/vam/project/rl-camera-v2/unity_camera_environment/unity_camera_v2/Assets/image/SavedScreen.jpg'
        self.config = './mmdetection/configs/yolox/yolox_s_8x8_300e_coco_camera.py'
        self.checkpoint = './mmdetection/yolox/weights/epoch_100.pth'
        self.class_name = ('car',)
        self.observed_target_features = []
        self.observed_target_img = []
        self.forward_observed_target_feature = []
        self.forward_observed_target_feature_list = []
        self.cfg = setup_cfg(args)
        self.feature_extractor = FeatureExtractionDemo(self.cfg, parallel=args.parallel)

    def reset(self):
        # Reset statistics
        sock.SendData("RESET,1")
        self.error_memory = []
        self.confidence_list = []
        self.action_time_list = []
        self.target_area_list = []
        self.cam_fov_list = []
        self.cam_fov_list.append(60.0)
        self.action_list = []
        self.observed_target_features = []
        self.observed_target_img = []
        self.forward_observed_target_feature = []
        self.forward_observed_target_feature_list = []
        self.current_center = [self.observation_size_w//2, self.observation_size_h//2]  #[1515, 1224]
        self.observation = np.zeros((self.observation_size_h, self.observation_size_w, 3,))
        self.detection_result = np.zeros((self.observation_size_h, self.observation_size_w, 3,))
        # Update statistics
        self.epoch_counter += 1
        self.iteration_counter = 0
        self.end_iteration = 0
        self.object_num = -1
        self.rtn_object_num = -1
        self.miss_target_num = 0
        self.object_speed = -1
        self.init_area_ratio = 0
        self.object_detect_num = 0
        self.envSeed = -2
        reward = 0
        self.error_memory.append(reward)
        # Reset text
        self.current_state_text = '{0:.4f}'.format(reward)
        self.action_text = 'None'
        self.action_time_text = 'None'
        self.forward_action = 0
        self.forward_action_time = 0
        self.forward_x = 0
        self.forward_y = 0
        self.forward_a = 0
        self.forward_c = 0
        self.forward_pos_punish = 0
        self.forward_area_punish = 0
        self.forward_con_punish = 0
        self.forward_inbox_punish = 0
        while 1:
            callback = sock.ReadReceivedData()
            # todo: 优化
            if callback is not None and "ActionDone" in callback:
                break
        load_img = None
        while load_img is None:
            try:
                load_img = cv2.imread(self.file_path)
            except:
                pass
        self.observation = load_img
        try:
            result = detect(self.file_path, self.config, self.checkpoint, out_file='./images/res.jpg')
        except:
            e = cv2.imread(self.file_path)
            cv2.imwrite("error.jpg", e)
            return [-2, -2, -1, 0, 0, 0, 0, 0]
        self.detection_result = self.observation.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]
        inds = scores > self.score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bbox_max = None
        label_max = -1
        bbox_max_idx = -1
        found = False
        found_forward_target = False
        if bboxes is not None:
            for i in range(0, bboxes.shape[0]):
                bbox = bboxes[i]
                label = labels[i]
                # find the max
                if label in [0]:
                    cv2.rectangle(self.detection_result, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]), int(bbox[3])),
                                  (255, 0, 0), 2)
                    x1, y1, x2, y2, c = bbox
                    bbox_max = bbox

        # Calculate and normalize reward
        if bbox_max is not None:
            temp_c = 1 - bbox_max[4]
            c = 1 - temp_c
            self.confidence_list.append(c)
            bbox_center_x = (bbox_max[2] - bbox_max[0]) / 2 + bbox[0]
            bbox_center_y = (bbox_max[3] - bbox_max[1]) / 2 + bbox[1]
            area_ratio = ((bbox_max[2] - bbox_max[0]) * (bbox_max[3] - bbox_max[1])) / (
                        self.observation_size_w * self.observation_size_h)
            self.target_area_list.append(area_ratio) # add
            if self.init_area_ratio == 0:
                self.init_area_ratio = area_ratio
            cv2.circle(self.detection_result, (int(bbox_center_x), int(bbox_center_y)), 2, (0, 0, 255), 2)
            cv2.circle(self.detection_result, (int(self.observation_size_w / 2), int(self.observation_size_h / 2)), 3,
                       (255, 0, 0), 3)
            cv2.arrowedLine(self.detection_result, (int(self.observation_size_w / 2), int(self.observation_size_h / 2)),
                            (int(bbox_center_x), int(bbox_center_y)), (255, 0, 0),
                            thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.2)
            x = (bbox_center_x - self.current_center[0]) / (self.observation_size_w / 2)
            y = (bbox_center_y - self.current_center[1]) / (self.observation_size_h / 2)
            a = area_ratio * 100
            if abs(bbox_center_x - self.current_center[0]) <= (bbox_max[2] - bbox_max[0]) / 2 and abs(
                    bbox_center_y - self.current_center[1]) <= (bbox_max[3] - bbox_max[1]) / 2:
                self.in_bbox = True
            else:
                self.in_bbox = False
        else:
            x = -2
            y = -2
            a = -1
            c = 0
        self.forward_action = 0
        self.forward_action_time = 0
        bbox_trend_x = 0
        bbox_trend_y = 0
        self.obs_vector = [x, y, a, c, self.forward_action, self.forward_action_time, bbox_trend_x, bbox_trend_y]
        return self.obs_vector

    def seed(self, seed):
        np.random.seed(seed)

    def _xywh_to_xyxy(self, bbox_xywh):
        # print(bbox_xywh)
        x, y, w, h, c= bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.observation_size_w - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.observation_size_h - 1)
        print(x,y,w,h)
        print("x1:{},y1:{},x2:{},y2:{}".format(x1, y1, x2, y2))
        return x1, y1, x2, y2

    def postprocess(self, features):
        # Normalize feature to compute cosine distance
        features = F.normalize(features)
        features = features.cpu().data.numpy()
        return features

    def return_zoom_error(self, area_ratio):
        if area_ratio >= 0.05:
            return np.exp(-(area_ratio - 0.05) / 0.1)
        elif area_ratio < 0.05:
            return 1 - (((area_ratio - 0.05) / 0.05) ** 2)

    def step(self, action):
        # Perform an action
        if action[0] == 0:
            sock.SendData("STAY" + ',' + str(action[1]))
            self.action_time_list.append(action[1]/5)
            self.action_list.append("STAY")
        elif action[0] == 1:  # down
            sock.SendData("DOWN" + ',' + str(action[2]))
            self.action_time_list.append(action[2]/5)
            self.action_list.append("DOWN")
        elif action[0] == 2:  # up
            sock.SendData("UP" + ',' + str(action[3]))
            self.action_time_list.append(action[3]/5)
            self.action_list.append("UP")
        elif action[0] == 3:  # right
            sock.SendData("RIGHT" + ',' + str(action[4]))
            self.action_time_list.append(action[4]/5)
            self.action_list.append("RIGHT")
        elif action[0] == 4:  # left
            sock.SendData("LEFT" + ',' + str(action[5]))
            self.action_time_list.append(action[5]/5)
            self.action_list.append("LEFT")
        elif action[0] == 5:  # zoom in
            sock.SendData("ZOOMIN" + ',' + str(action[6]))
            self.action_time_list.append(action[6]/5)
            self.action_list.append("ZOOMIN")
        elif action[0] == 6:  # zoom out
            sock.SendData("ZOOMOUT" + ',' + str(action[7]))
            self.action_time_list.append(action[7]/5)
            self.action_list.append("ZOOMOUT")
        else:
            assert False
        # add the code for receive Unity callback
        while 1:
            callback = sock.ReadReceivedData()
            # todo: 优化
            if callback is not None and "ActionDone" in callback:
                # 使用split()方法拆分字符串
                parts = callback.split('|')
                # 获取|之后的数字并转换为整数
                if len(parts) == 2:
                    if self.object_detect_num == 0:
                        self.object_num = int(parts[1].split('#')[0])
                        self.rtn_object_num = int(parts[1].split('#')[0])
                # 使用split()方法拆分字符串
                parts = callback.split('#')
                # 获取|之后的数字并转换为整数
                if len(parts) == 2:
                    if self.object_speed == -1:
                        self.object_speed = float(parts[1].split('%')[0])
                # 使用split()方法拆分字符串
                parts = callback.split('%')
                # 获取|之后的数字并转换为整数
                if len(parts) == 2:
                    if self.envSeed == -2:
                        self.envSeed = int(parts[1].split('*')[0])# int(parts[1])
                # # 使用split()方法拆分字符串
                # parts = callback.split('*') # add
                # # 获取|之后的数字并转换为整数
                # if len(parts) == 2:
                #     self.cam_fov_list.append(float(parts[1]))
                else:
                    print("object_num error ")
                break
        self.observation = cv2.imread(self.file_path)
        try:
            result = detect(self.file_path, self.config, self.checkpoint, out_file='./images/res.jpg')
        except:
            e = cv2.imread(self.file_path)
            cv2.imwrite("error.jpg", e)
            return [-2, -2, -1, 0, 0, 0, 0, 0], 0, False, {}
        self.detection_result = self.observation.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        scores = bboxes[:, -1]
        inds = scores > self.score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        bbox_max = None
        label_max = -1
        bbox_max_idx = -1
        found = False
        found_forward_target = False
        if bboxes is not None:
            for i in range(0, bboxes.shape[0]):
                bbox = bboxes[i]
                label = labels[i]
                # find the max
                if label in [0]:
                    x1, y1, x2, y2, c = bbox
                    x1 = max(int(x1), 0)
                    x2 = min(int(x2), self.observation_size_w - 1)
                    y1 = max(int(y1), 0)
                    y2 = min(int(y2), self.observation_size_h - 1)
                    im = self.observation[y1:y2, x1:x2]
                    if im is None:
                        continue
                    else:
                        cv2.imshow('Select bbox', im)
                    feat = self.feature_extractor.run_on_image(im)
                    feature_select = self.postprocess(feat)
                    if len(self.forward_observed_target_feature) is not 0:
                        if (1-cosine(feature_select, self.forward_observed_target_feature)) < self.THRESHOLD:
                            continue
                        else:
                            self.forward_observed_target_feature = feature_select*0.85 + self.forward_observed_target_feature*0.15
                            self.forward_observed_target_feature_list.append(feature_select)
                            found_forward_target = True
                    else:
                        self.forward_observed_target_feature = feature_select
                        found_forward_target = True
                    for n in range(0, len(self.observed_target_features)):
                        similarity = 1 - cosine(feature_select, self.observed_target_features[n])
                        # 统一缩放图像大小到指定尺寸
                        scaled_select_images = cv2.resize(im, (200, 100))
                        scaled_target_images = cv2.resize(self.observed_target_img[n], (200, 100))
                        imgs = np.hstack([scaled_select_images, scaled_target_images])
                        cv2.putText(imgs, str(similarity), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)
                        cv2.imshow("Compare Similarity", imgs)
                        if similarity > self.THRESHOLD:
                            found_forward_target = False
                            self.forward_observed_target_feature = []
                            self.forward_observed_target_feature_list = []
                            break
                    else:
                        found = True
                    if found:
                        bbox_max = bbox
                        label_max = label
                        bbox_deque.append(bbox_max)
                        cv2.rectangle(self.detection_result, (int(bbox_max[0]), int(bbox_max[1])),
                                      (int(bbox_max[2]), int(bbox_max[3])),
                                      (0, 0, 255), 2)
                        break


        if not found_forward_target:
            self.forward_observed_target_feature = []
            self.forward_observed_target_feature_list = []

        if action[0] == 0:
            action_text = 'Stay'
            action_time_text = str(action[1])
        elif action[0] == 1:
            action_text = 'Down'
            action_time_text = str(action[2])
        elif action[0] == 2:
            action_text = 'Up'
            action_time_text = str(action[3])
        elif action[0] == 3:
            action_text = 'Right'
            action_time_text = str(action[4])
        elif action[0] == 4:
            action_text = 'Left'
            action_time_text = str(action[5])
        elif action[0] == 5:
            action_text = 'Zoom in'
            action_time_text = str(action[6])
        elif action[0] == 6:
            action_text = 'Zoom out'
            action_time_text = str(action[7])
        # Calculate and normalize reward
        if bbox_max is not None:
            con_punish = 1 - bbox_max[4]  # the highest confidence of the finding object
            c = 1 - con_punish
            self.confidence_list.append(c)
            bbox_center_x = (bbox_max[2] - bbox_max[0])/2 + bbox[0]
            bbox_center_y = (bbox_max[3] - bbox_max[1])/2 + bbox[1]
            area_ratio = ((bbox_max[2] - bbox_max[0]) * (bbox_max[3] - bbox_max[1]))/(self.observation_size_w * self.observation_size_h)
            self.target_area_list.append(area_ratio) # add
            if self.init_area_ratio == 0:
                print(area_ratio)
                self.init_area_ratio = area_ratio
            cv2.circle(self.detection_result, (int(bbox_center_x), int(bbox_center_y)), 2, (0, 0, 255), 2)
            cv2.circle(self.detection_result, (int(self.observation_size_w/2), int(self.observation_size_h/2)), 3, (255, 0, 0), 3)
            cv2.arrowedLine(self.detection_result, (int(self.observation_size_w/2), int(self.observation_size_h/2)), (int(bbox_center_x), int(bbox_center_y)), (255, 0, 0),
                            thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.2)
            x = (bbox_center_x - self.current_center[0])/(self.observation_size_w/2)
            y = (bbox_center_y - self.current_center[1])/(self.observation_size_h/2)
            a = area_ratio * 100
            if abs(bbox_center_x - self.current_center[0]) <= (bbox_max[2] - bbox_max[0])/2 and abs(bbox_center_y - self.current_center[1]) <= (bbox_max[3] - bbox_max[1])/2:
                self.in_bbox = True
            else:
                self.in_bbox = False
            pos_punish = float(1/2) * (((bbox_center_x - self.observation_size_w/2)/(self.observation_size_w/2))**2
                                        + ((bbox_center_y - self.observation_size_h/2)/(self.observation_size_h/2))**2)
            area_punish = 1 - self.return_zoom_error(area_ratio)
            step_punish = 1 - ((-1 / 512) * self.iteration_counter + 1)
            # 奖励目标保持在视野中心
            reward = - (con_punish + pos_punish + area_punish)
            if self.in_bbox:
                in_bbox_punish = 0
                reward += in_bbox_punish
                cv2.circle(self.detection_result, (int(self.observation_size_w / 2), int(self.observation_size_h / 2)),
                           3, (0, 255, 0), 3)
            else:
                in_bbox_punish = -1
                reward += in_bbox_punish
        else:
            x = -2
            y = -2
            a = -1
            c = 0
            self.in_bbox = False
            in_bbox_punish = -1
            con_punish = 1
            pos_punish = 1
            step_punish = 1
            area_punish = 1
            reward = - (con_punish + pos_punish + area_punish)
            reward += in_bbox_punish
            self.miss_target_num += 1

        self.error_memory.append(reward)

        if self.forward_pos_punish - pos_punish > 0.2 and self.forward_pos_punish != 1:
            reward += 0.5
        if self.forward_area_punish - area_punish > 0.15 and self.forward_area_punish != 1:
            reward += 0.5
        if self.forward_con_punish - con_punish > 0.1 and self.forward_con_punish != 1:
            reward += 0.5
        if in_bbox_punish > self.forward_inbox_punish and self.forward_inbox_punish != -1:
            reward += 0.5

        self.forward_pos_punish = pos_punish
        self.forward_area_punish = area_punish
        self.forward_con_punish = con_punish
        self.forward_inbox_punish = in_bbox_punish

        # Reset text
        self.current_state_text = 'reward:{0:.4f}'.format(reward) + ' ' + 'r_p:{0:.4f}'.format(pos_punish) + ' ' + 'r_c:{0:.4f}'.format(con_punish) + ' ' + 'r_i:{0:.4f}'.format(in_bbox_punish)
        self.action_text = action_text
        self.action_time_text = action_time_text
        if (1 - con_punish) > 0.9:
            self.end_iteration += 1
        else:
            self.end_iteration = 0

        self.iteration_counter += 1
        done = False
        if self.iteration_counter > 127:  # end_iteration = 8
            done = True
            reward -= 40
        elif self.end_iteration >= 10:
            # done condition should change ,like time?
            if bbox_max is not None:
                im_max_crops = []
                x1, y1, x2, y2, c = bbox_max
                x1 = max(int(x1), 0)
                x2 = min(int(x2), self.observation_size_w - 1)
                y1 = max(int(y1), 0)
                y2 = min(int(y2), self.observation_size_h - 1)
                im = self.observation[y1:y2, x1:x2]
                if im is not None:
                    f = np.mean(self.forward_observed_target_feature_list, axis=0)
                    self.observed_target_features.append(f)
                    self.observed_target_img.append(im)
                self.forward_observed_target_feature = []
                self.forward_observed_target_feature_list = []
            done = False
            self.end_iteration = 0
            self.object_num -= 1
            self.object_detect_num += 1
            reward += 4
        if self.object_num == 0:
            done = True
            reward += self.object_detect_num * 40

        if self.forward_x==-2 and self.forward_y==-2:
            bbox_trend_x = 0
            bbox_trend_y = 0
        else:
            bbox_trend_x = x - self.forward_x
            bbox_trend_y = y - self.forward_y
        if bbox_max is not None:
            self.obs_vector = [x, y, a, c, self.forward_action, self.forward_action_time, bbox_trend_x, bbox_trend_y]
        else:
            self.obs_vector = [-2, -2, -1, 0, 0, 0, 0, 0]
        self.forward_x = x
        self.forward_y = y
        self.forward_a = a
        self.forward_c = c
        if bbox_max is not None:
            self.forward_action = action[0] / 6
            self.forward_action_time = action[int(self.forward_action)] / 5  # action time 归一化
        else:
            self.forward_action = 0
            self.forward_action_time = 0
        return self.obs_vector, reward, done, {}

    def render(self, mode='Car', close=False):   # Default

        cv2.putText(self.detection_result, "State: " + self.current_state_text, (5, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Sel. Action: " + self.action_text, (5, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Action Time: " + self.action_time_text, (5, 150), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Seek Object: " + mode, (5, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Vector: " + "[" + str(round(self.obs_vector[0], 2)) +", " + str(round(self.obs_vector[1], 2)) + "]", (5, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0),3)
        cv2.putText(self.detection_result, "Area Ratio: " + "[" + str(round(self.obs_vector[2], 5)) + "]", (5, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Confidence: " + "[" + str(round(self.obs_vector[3], 2)) + "]", (5, 350),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "Action before: " + "[" + str(round(self.obs_vector[4], 2)) + "]", (5, 400),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.detection_result, "action_counter:" + str(self.iteration_counter), (3, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 3)
        cv2.namedWindow('Detection Result', 0)
        cv2.resizeWindow('Detection Result', 960, 540)  # 自己设定窗口图片的大小
        cv2.imshow('Detection Result', self.detection_result)
        # if need save the image list
        # img_list_path = 'PDQN/image_list/'
        # cv2.imwrite(img_list_path + str(self.iteration_counter) + '.jpg', self.detection_result)
        cv2.waitKey(10)



    def interactive_keyboard(self):
        if self.interactive:
            key = cv2.waitKey(0)
            return key
        else:
            cv2.waitKey(10)
            return 0

    def __str__(self):
        return 'UnityCameraControlEnvironment'

if __name__=='__main__':

    keyboard = {'w': 119, 's': 115, 'a': 97, 'd': 100, 'q': 113, 'e': 101}

    np.random.seed(1)
    env = UnityCameraControlEnv(a_p=0, a_r=0, e_thres=0, interactive=True)
    env.seed(1)
    env.reset()
    action = 0
    for i in range(1000):
        key = env.interactive_keyboard()
        if key == keyboard['w']:  # 'Up'
            action = 2
        elif key == keyboard['s']:  # 'Down'
            action = 1
        elif key == keyboard['a']:  # 'Left'
            action = 4
        elif key == keyboard['d']:  # 'Right'
            action = 3
        elif key == keyboard['q']:  # 'Zoom in'
            action = 5
        elif key == keyboard['e']:  # 'Zoom out'
            action = 6
        else:
            action = 0
        env.step(action)
        env.render()
    env.close()

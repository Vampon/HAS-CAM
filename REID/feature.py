import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from nn_matching import NearestNeighborDistanceMetric,_nn_euclidean_distance
from model import Net
# from model2 import Net
# from fastreid.config import get_cfg
# from fastreid.engine import DefaultTrainer
# from fastreid.utils.checkpoint import Checkpointer
from scipy.spatial.distance import cosine
import os

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True).cuda()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (128, 64)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


# class FastReIDExtractor(object):
#     def __init__(self, model_config, model_path, use_cuda=True):
#         cfg = get_cfg()
#         cfg.merge_from_file(model_config)
#         cfg.MODEL.BACKBONE.PRETRAIN = False
#         self.net = DefaultTrainer.build_model(cfg)
#         self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
#
#         Checkpointer(self.net).load(model_path)
#         logger = logging.getLogger("root.tracker")
#         logger.info("Loading weights from {}... Done!".format(model_path))
#         self.net.to(self.device)
#         self.net.eval()
#         height, width = cfg.INPUT.SIZE_TEST
#         self.size = (width, height)
#         self.norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#
#     def _preprocess(self, im_crops):
#         def _resize(im, size):
#             return cv2.resize(im.astype(np.float32) / 255., size)
#
#         im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
#         return im_batch
#
#     def __call__(self, im_crops):
#         im_batch = self._preprocess(im_crops)
#         with torch.no_grad():
#             im_batch = im_batch.to(self.device)
#             features = self.net(im_batch)
#         return features.cpu().numpy()

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def resize(im, size):
    return cv2.resize(im.astype(np.float32) / 255., size)

if __name__ == '__main__':
    # img1 = cv2.imread("car/1.jpg")
    # img2 = cv2.imread("car/11.jpg")
    # im_crops = []
    # im_crops.append(img1)
    # im_crops.append(img2)
    #
    # img3 = cv2.imread("car/2.jpg")
    # img4 = cv2.imread("car/31.jpg")
    # im_crops2 = []
    # im_crops2.append(img3)
    # im_crops2.append(img4)
    # extr = Extractor("checkpoint/ckpt.t7")
    #
    # max_cosine_distance = 0.2
    # nn_budget = 100
    # metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # fea = extr(im_crops)
    # fea2 = extr(im_crops2)
    # print(fea.shape)
    # # sim = cosin_metric(fea[0], fea[1])
    # sim = _nn_euclidean_distance(fea, fea2)
    # print(sim)

    # # 读取图像
    # img1 = cv2.imread("car/3.jpg")
    # img2 = cv2.imread("car/1.jpg")
    #
    # # 提取特征向量
    # im_crops = [img1, img2]
    # extr = Extractor("checkpoint/ckpt.t7")
    # features = extr(im_crops)
    #
    # # 计算余弦相似度
    # similarity = 1 - cosine(features[0], features[1])
    #
    # print("图像相似度:", similarity)


    # 文件夹路径
    folder_path = "car"

    # 读取文件夹内的图像
    image_files = os.listdir(folder_path)

    # 提取特征向量
    im_crops = []
    extr = Extractor("checkpoint/ckpt.t7")
    features = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        im_crops.append(img)
        # feature = extr([img])
        # features.append(feature)
    features = extr(im_crops)
    # 计算相似度矩阵
    num_images = len(image_files)
    similarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i + 1, num_images):
            similarity = 1 - cosine(features[i], features[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # 展示相似度结果
    for i in range(num_images):
        for j in range(i + 1, num_images):
            image1 = image_files[i]
            image2 = image_files[j]
            similarity = similarity_matrix[i, j]
            print(f"相似度({image1}, {image2}): {similarity}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage.measurements import label

class Bin_Mean_Shift(nn.Module):
    def __init__(self, train_iter=5, test_iter=10, bandwidth=0.5, device='cpu'):
        super(Bin_Mean_Shift, self).__init__()
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.bandwidth = bandwidth / 2.
        self.anchor_num = 10
        self.sample_num = 3000
        self.device = device

    def generate_seed(self, point, bin_num):
        """
        :param point: tensor of size (K, 2)
        :param bin_num: int
        :return: seed_point
        """
        def get_start_end(a, b, k):
            start = a + (b - a) / ((k + 1) * 2)
            end = b - (b - a) / ((k + 1) * 2)
            return start, end

        min_x, min_y = point.min(dim=0)[0]
        max_x, max_y = point.max(dim=0)[0]

        start_x, end_x = get_start_end(min_x.item(), max_x.item(), bin_num)
        start_y, end_y = get_start_end(min_y.item(), max_y.item(), bin_num)

        x = torch.linspace(start_x, end_x, bin_num).view(bin_num, 1)
        y = torch.linspace(start_y, end_y, bin_num).view(1, bin_num)

        x_repeat = x.repeat(1, bin_num).view(-1, 1)
        y_repeat = y.repeat(bin_num, 1).view(-1, 1)

        return torch.cat((x_repeat, y_repeat), dim=1).to(self.device)

    def filter_seed(self, point, prob, seed_point, bandwidth, min_count=3):
        """
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param min_count:  mini_count within a bandwith of seed point
        :param bandwidth: float
        :return: filtered_seed_points
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        thres_matrix = (distance_matrix < bandwidth).type(torch.float32) * prob.t()
        count = thres_matrix.sum(dim=1)                  # (n, 1)
        valid = count > min_count
        return seed_point[valid]

    def cal_distance_matrix(self, point_a, point_b):
        """
        :param point_a: tensor of size (m, 2)
        :param point_b: tensor of size (n, 2)
        :return: distance matrix of size (m, n)
        """
        m, n = point_a.size(0), point_b.size(0)

        a_repeat = point_a.repeat(1, n).view(n * m, 2)                  # (n*m, 2)
        b_repeat = point_b.repeat(m, 1)                                 # (n*m, 2)

        distance = torch.nn.PairwiseDistance(keepdim=True)(a_repeat, b_repeat)  # (n*m, 1)

        return distance.view(m, n)

    def shift(self, point, prob, seed_point, bandwidth):
        """
        shift seed points
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param bandwidth: float
        :return:  shifted points with size (n, 2)
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        kernel_matrix = torch.exp((-0.5 / bandwidth**2) * (distance_matrix ** 2)) * (1. / (bandwidth * np.sqrt(2 * np.pi)))
        weighted_matrix = kernel_matrix * prob.t()

        # normalize matrix
        normalized_matrix = weighted_matrix / weighted_matrix.sum(dim=1, keepdim=True)
        shifted_point = torch.matmul(normalized_matrix, point)  # (n, K) * (K, 2) -> (n, 2)

        return shifted_point

    def label2onehot(self, labels):
        """
        convert a label to one hot vector
        :param labels: tensor with size (n, 1)
        :return: one hot vector tensor with size (n, max_lales+1)
        """
        n = labels.size(0)
        label_num = torch.max(labels).int() + 1

        onehot = torch.zeros((n, label_num))
        onehot.scatter_(1, labels.long(), 1.)

        return onehot.to(self.device)

    def merge_center(self, seed_point, bandwidth=0.25):
        """
        merge close seed points
        :param seed_point: tensor of size (n, 2)
        :param bandwidth: float
        :return: merged center
        """
        n = seed_point.size(0)

        # 1. calculate intensity
        distance_matrix = self.cal_distance_matrix(seed_point, seed_point)  # (n, n)
        intensity = (distance_matrix < bandwidth).type(torch.float32).sum(dim=1)

        # merge center if distance between two points less than bandwidth
        sorted_intensity, indices = torch.sort(intensity, descending=True)
        is_center = np.ones(n, dtype=np.bool)
        indices = indices.cpu().numpy()
        center = np.zeros(n, dtype=np.uint8)

        labels = np.zeros(n, dtype=np.int32)
        cur_label = 0
        for i in range(n):
            if is_center[i]:
                labels[indices[i]] = cur_label
                center[indices[i]] = 1
                for j in range(i + 1, n):
                    if is_center[j]:
                        if distance_matrix[indices[i], indices[j]] < bandwidth:
                            is_center[j] = 0
                            labels[indices[j]] = cur_label
                cur_label += 1
        # print(labels)
        # print(center)
        # return seed_point[torch.ByteTensor(center)]

        # change mask select to matrix multiply to select points
        one_hot = self.label2onehot(torch.Tensor(labels).view(-1, 1))  # (n, label_num)
        weight = one_hot / one_hot.sum(dim=0, keepdim=True)   # (n, label_num)

        return torch.matmul(weight.t(), seed_point)

    def cluster(self, point, center):
        """
        cluter each point to nearset center
        :param point: tensor with size (K, 2)
        :param center: tensor with size (n, 2)
        :return: clustering results, tensor with size (K, n) and sum to one for each row
        """
        # plus 0.01 to avoid divide by zero
        distance_matrix = 1. / (self.cal_distance_matrix(point, center)+0.01)  # (K, n)
        segmentation = F.softmax(distance_matrix, dim=1)
        return segmentation

    def bin_shift(self, prob, embedding, param, gt_seg, bandwidth):
        """
        discrete seeding mean shift in training stage
        :param prob: tensor with size (1, h, w) indicate probability of being plane
        :param embedding: tensor with size (2, h, w)
        :param param: tensor with size (3, h, w)
        :param gt_seg: ground truth instance segmentation, used for sampling planar embeddings
        :param bandwidth: float
        :return: segmentation results, tensor with size (h*w, K), K is cluster number, row sum to 1
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                center, tensor with size (K, 2) cluster center in embedding space
                sample_prob, tensor with size (N, 1) sampled probability
                sample_seg, tensor with size (N, 1) sampled ground truth instance segmentation
                sample_params, tensor with size (3, N), sampled params
        """

        c, h, w = embedding.size()

        embedding = embedding.view(c, h*w).t()
        param = param.view(3, h*w)
        prob = prob.view(h*w, 1)
        seg = gt_seg.view(-1)

        # random sample planar region data points using ground truth label to speed up training
        rand_index = np.random.choice(np.arange(0, h * w)[seg.cpu().numpy() != 20], self.sample_num)

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, rand_index]

        # generate seed points and filter out those with low density to speed up training
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None

        with torch.no_grad():
            for iter in range(self.train_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points
        segmentation = self.cluster(embedding, center)
        sampled_segmentation = segmentation[rand_index]

        return segmentation, sampled_segmentation, center, sample_prob, seg[rand_index].view(-1, 1), sample_param

    def forward(self, logit, embedding, param, gt_seg):
        batch_size, c, h, w = embedding.size()
        assert(c == 2)

        # apply mean shift to every item
        segmentations, sample_segmentations, centers, sample_probs, sample_gt_segs, sample_params = [], [], [], [], [], []
        for b in range(batch_size):
            segmentation, sample_segmentation, center, prob, sample_seg, sample_param = \
                self.bin_shift(torch.sigmoid(logit[b]), embedding[b], param[b], gt_seg[b], self.bandwidth)

            segmentations.append(segmentation)
            sample_segmentations.append(sample_segmentation)
            centers.append(center)
            sample_probs.append(prob)
            sample_gt_segs.append(sample_seg)
            sample_params.append(sample_param)

        return segmentations, sample_segmentations, sample_params, centers, sample_probs, sample_gt_segs

    def test_forward(self, prob, embedding, mask_threshold):
        """
        :param prob: probability of planar, tensor with size (1, h, w)
        :param embedding: tensor with size (2, h, w)
        :param mask_threshold: threshold of planar region
        :return: clustering results: numpy array with shape (h, w),
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                 sample_params, tensor with size (3, N), sampled params
        """

        c, h, w = embedding.size()

        embedding = embedding.view(c, h*w).t()
        prob = prob.view(h*w, 1)

        # random sample planar region data points
        rand_index = np.random.choice(np.arange(0, h * w)[prob.cpu().numpy().reshape(-1) > mask_threshold], self.sample_num)

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]

        # generate seed points and filter out those with low density
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)

        with torch.no_grad():
            # start shift points
            for iter in range(self.test_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points using sample_embedding
        segmentation = self.cluster(embedding, center)

        sampled_segmentation = segmentation[rand_index]

        return segmentation, sampled_segmentation


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


colors = labelcolormap(256)

def gen_segmentation(bin_mean_shift, prob, embedding, mask_threshold=0.1, return_sep_plane=False, planeAreaThreshold=1000):
    """
    :param prob: probability of planar, tensor with size (1, h, w)
    :param embedding: tensor with size (2, h, w)
    :param mask_threshold: threshold of planar region
    """
    import time
    _, h, w = prob.shape
    segmentation, sampled_segmentation = bin_mean_shift.test_forward(
        prob, embedding, mask_threshold=mask_threshold)
    # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
    # we thus use avg_pool_2d to smooth the segmentation results
    b = segmentation.t().view(1, -1, h, w)
    pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
    b = pooling_b.view(-1, h * w).t()
    segmentation = b

    # return cluster results
    predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

    # mask out non planar region
    predict_segmentation[prob.cpu().numpy().reshape(-1) <= mask_threshold] = 20
    predict_segmentation = predict_segmentation.reshape(h, w)

    # TODO: continue to tune the plane seperation
    # plane_mask, non_plane_mask = read_plane_seg_self_conneted(plane_seg_filepath='', planeAreaThreshold=1000, seg=predict_segmentation)
    if return_sep_plane:
        plane_mask, non_plane_mask = read_plane_seg_self_conneted(plane_seg_filepath='', planeAreaThreshold=planeAreaThreshold, seg=predict_segmentation)
    else:
        plane_mask, non_plane_mask = read_plane_seg_self(plane_seg_filepath='', planeAreaThreshold=planeAreaThreshold, seg=predict_segmentation)
    # # visualization and evaluation
    # image = tensor_to_image(image.cpu()[0])
    # mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)

    # change non planar to zero, so non planar region use the black color
    # predict_segmentation += 1
    # predict_segmentation[predict_segmentation == 21] = 0
    predict_segmentation = np.concatenate([non_plane_mask[np.newaxis], plane_mask]).reshape(-1, h*w).argmax(axis=0).reshape(h, w)
    seg_vis_img = np.stack([colors[predict_segmentation, 0], colors[predict_segmentation, 1], colors[predict_segmentation, 2]], axis=2)

    return plane_mask, non_plane_mask, seg_vis_img

    # # blend image
    # blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)
    #
    # mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def gen_segmentation_infer(bin_mean_shift, prob, embedding, mask_threshold=0.1, return_sep_plane=False):
    """
    :param prob: probability of planar, tensor with size (1, h, w)
    :param embedding: tensor with size (2, h, w)
    :param mask_threshold: threshold of planar region
    """
    _, h, w = prob.shape

    segmentation, sampled_segmentation = bin_mean_shift.test_forward(
        prob, embedding, mask_threshold=mask_threshold)

    # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned,
    # we thus use avg_pool_2d to smooth the segmentation results
    b = segmentation.t().view(1, -1, h, w)
    pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
    b = pooling_b.view(-1, h * w).t()
    segmentation = b

    if return_sep_plane:
        # return cluster results
        predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)
        predNumPlanes = len(np.unique(predict_segmentation)) - 1
        if predNumPlanes > 20:
            print('predNumPlanes > 20')

        # mask out non planar region
        predict_segmentation[prob.cpu().numpy().reshape(-1) <= mask_threshold] = 20
        predict_segmentation = predict_segmentation.reshape(h, w)

        plane_mask_sp, non_plane_mask_sp = read_plane_seg_self_conneted(plane_seg_filepath='', planeAreaThreshold=1000, seg=predict_segmentation)
        plane_mask_sp = plane_mask_sp.astype(np.float).reshape(-1, h*w).transpose()
        return torch.from_numpy(plane_mask_sp).to(segmentation.device)

    return segmentation

    # # return cluster results
    # predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)
    #
    # # mask out non planar region
    # predict_segmentation[prob.cpu().numpy().reshape(-1) <= mask_threshold] = 20
    # predict_segmentation = predict_segmentation.reshape(h, w)
    #
    # # TODO: continue to tune the plane seperation
    # plane_mask, non_plane_mask = read_plane_seg_self(plane_seg_filepath='', planeAreaThreshold=1000, seg=predict_segmentation)
    # if return_sep_plane:
    #     plane_mask_sp, non_plane_mask_sp = read_plane_seg_self_conneted(plane_seg_filepath='', planeAreaThreshold=1000, seg=predict_segmentation)
    #
    # # # visualization and evaluation
    # # image = tensor_to_image(image.cpu()[0])
    # # mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
    #
    # # change non planar to zero, so non planar region use the black color
    # predict_segmentation += 1
    # predict_segmentation[predict_segmentation == 21] = 0
    #
    # seg_vis_img = np.stack([colors[predict_segmentation, 0], colors[predict_segmentation, 1], colors[predict_segmentation, 2]], axis=2)
    #
    # if return_sep_plane:
    #     return plane_mask, non_plane_mask, seg_vis_img, plane_mask_sp, non_plane_mask_sp
    #
    # return plane_mask, non_plane_mask, seg_vis_img


def read_plane_seg_gt_simple(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    if np.max(seg) >= 20:
        raise NotImplementedError

    seg[seg == np.max(seg)] = 20  # non-planar set to 20

    ori_indices, counts = np.unique(seg, return_counts=True)
    plane_num = len(ori_indices) - 1

    instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, non_plane_mask


def read_plane_seg_gt_eval(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    if np.max(seg) >= 20:
        raise NotImplementedError

    seg[seg == np.max(seg)] = 20  # non-planar set to 20

    ori_indices, counts = np.unique(seg, return_counts=True)
    plane_num = len(ori_indices) - 1

    instance = np.zeros([21, seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            instance[20] = seg == ori_index
            assert np.sum(instance[20]) > 0
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, plane_num


def read_plane_seg_self(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    # if np.max(seg) >= 20:
    #     raise NotImplementedError

    # seg[seg == np.max(seg)] = 20  # non-planar set to 20
    ori_indices, counts = np.unique(seg, return_counts=True)
    if np.max(seg) == 20:
        plane_num = len(ori_indices) - 1
    else:
        plane_num = len(ori_indices)

    instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)

    new_index = 0
    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance[new_index] = seg == ori_index
        new_index += 1

    return instance, non_plane_mask


def read_plane_seg_self_conneted(plane_seg_filepath, planeAreaThreshold=200, seg=None):
    if seg is None:
        seg = cv2.imread(plane_seg_filepath, -1)

    # planeAreaThreshold = 500

    # if np.max(seg) >= 20:
    #     raise NotImplementedError

    # seg[seg == np.max(seg)] = 20  # non-planar set to 20
    # seg: 480 x 640
    # TODO: get connected component and remove small plane regions

    ori_indices, counts = np.unique(seg, return_counts=True)

    # instance = np.zeros([plane_num, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    # instance = np.zeros([1, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    non_plane_mask = np.zeros([seg.shape[0], seg.shape[1]], dtype=np.uint8)
    instances = []
    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter

    for ori_index in ori_indices:
        if ori_index == 20:
            non_plane_mask = seg == ori_index
            assert np.sum(non_plane_mask) > 0
            continue

        if np.sum(seg == ori_index) < planeAreaThreshold:
            continue

        instance_all = seg == ori_index
        labeled, ncomponents = label(instance_all, structure)
        for i in range(1, ncomponents+1):
            instance_i = labeled == i
            if np.sum(instance_i) < planeAreaThreshold:
                continue

            instance_range = np.nonzero(instance_i)
            min_row = instance_range[0].min()
            max_row = instance_range[0].max()
            min_col = instance_range[1].min()
            max_col = instance_range[1].max()
            range_row = max_row - min_row
            range_col = max_col - min_col
            if range_row < 50 or range_col < 50:
                continue

            instances.append(instance_i)

        # instance[new_index] = seg == ori_index
        # new_index += 1

    if len(instances) > 0:
        instance = np.stack(instances, axis=0)
    else:
        instance = np.zeros([0, seg.shape[0], seg.shape[1]], dtype=np.uint8)
    return instance, non_plane_mask
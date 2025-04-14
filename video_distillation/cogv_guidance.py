from typing import Union, List
from jaxtyping import Float, Int

import os
import cv2

import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torch import Tensor
from threestudio_utils import get_device

FLOW_HEIGHT = 256
FLOW_WIDTH = 480

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[None, ...]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def preprocess(batch, height=FLOW_HEIGHT, width=FLOW_WIDTH, dtype=torch.float32):
    batch = (batch * 2. - 1.).to(dtype).clamp(-1.0, 1.0)
    batch = F.interpolate(
            batch, (height, width), mode="bilinear", align_corners=False
        )
    return batch


class CogVideoGuidance:
    def __init__(self, guidance_path, downsample=1., num_frames=8):
        self.device = get_device()
        self.weights_dtype = torch.float32
        self.FLOW_HEIGHT = 256
        self.FLOW_WIDTH = 480
        
        # load frames
        fnames = os.listdir(guidance_path)
        fnames.sort()
        frame_per_stage = len(fnames) // num_frames
        
        frames = []
        for i, fn in enumerate(fnames):
            if i % frame_per_stage != 0:
                continue
            frame = cv2.imread(os.path.join(guidance_path, fn))
            if downsample != 1.:
                frame = cv2.resize(frame, (frame.shape[1]*downsample, frame.shape[0]*downsample)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = numpy_to_pt(frame / 255.)
            frames.append(frame)
        self.guidance = torch.cat(frames, dim=0).to(self.device) # (BCHW)
        print("guidance shape:",self.guidance.shape)

        # predict optical flow
        prev_batch = preprocess(self.guidance[:-1])
        curr_batch = preprocess(self.guidance[1:])
        self.model = raft_large(pretrained=True, progress=False).to(self.device)
        self.model = self.model.eval()

        with torch.no_grad():
            self.guidance_flows = self.model(prev_batch, curr_batch)[-1]
        
        # load masks -- PAC-NeRF
        # self.guidance_masks = None
        # mask_path = guidance_path.replace(guidance_path.split('/')[-1], 'masks')
        # if os.path.exists(mask_path):
        #     fnames = os.listdir(mask_path)
        #     fnames.sort()
            
        #     frames = []
        #     for i, fn in enumerate(fnames):
        #         if i % frame_per_stage != 0:
        #             continue
        #         frame = cv2.imread(os.path.join(mask_path, fn), cv2.IMREAD_UNCHANGED)
        #         if downsample != 1.:
        #             frame = cv2.resize(frame, (frame.shape[1]*downsample, frame.shape[0]*downsample)) 
        #         frame = numpy_to_pt(frame[..., -1:] / 255.)
        #         frames.append(frame)
        #     self.guidance_masks = torch.cat(frames, dim=0).to(self.device) # (BCHW)
        #     print("guidance_masks shape:",self.guidance_masks.shape)


    def predict_flow(self, rgb_BCHW, rgb_BCHW_init):
        # predict optical flow
        prev_batch = preprocess(torch.concat([rgb_BCHW_init, rgb_BCHW[:-1]], dim=0)).detach()
        curr_batch = preprocess(rgb_BCHW)

        rgb_flows = self.model(prev_batch, curr_batch)[-1]
        ## debug
        flow_imgs = flow_to_image(rgb_flows)

        return rgb_flows, flow_imgs

    def normalize_flow(self, flow):
        max_norm = torch.sum(flow**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flow).dtype).eps
        normalized_flow = flow / (max_norm + epsilon)
        return normalized_flow

    def __call__(
        self,
        rgb_BCHW: Float[Tensor, "B C H W"],
        rgb_BCHW_init: Float[Tensor, "B C H W"],
        bbox_2d=None,
        reduction="sum"
    ):
        rgb_flows, _ = self.predict_flow(rgb_BCHW, rgb_BCHW_init)

        loss = {}
        if bbox_2d is None:
            bbox_2d = [0, rgb_BCHW.shape[2], 0, rgb_BCHW.shape[3]]
        
        bbox_2d = np.asarray(bbox_2d)
        bbox_2d[:2] = bbox_2d[:2] / rgb_BCHW.shape[2] * self.FLOW_HEIGHT
        bbox_2d[2:] = bbox_2d[2:] / rgb_BCHW.shape[3] * self.FLOW_WIDTH

        loss_flow = 0.5 * F.mse_loss(rgb_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]],
                                        self.guidance_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]], 
                                        reduction=reduction)
        loss['loss_flow'] = loss_flow

        return loss
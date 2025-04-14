#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_width, image_height,
                 image_path, image_name, uid, 
                 image=None, gt_alpha_mask=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 preload_img=True, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.nearest_id = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        if 'Courthouse' in image_path or 'bicycle' in image_path or 'dtu' in image_path or \
            'treehill' in image_path or 'stump' in image_path or 'flowers' in image_path or 'campus' in image_path:
            self.data_device = torch.device("cpu")

        self.original_image, self.image_gray, self.sky_mask = None, None, None
        self.preload_img = preload_img
        if self.preload_img:
            if 'co3d' in self.image_path:
                self.image_path = self.image_path[:-4]+'.jpg'
                if not os.path.exists(self.image_path):
                    self.image_path = self.image_path[:-4]+'.png'
            image = Image.open(self.image_path)
            resized_image = image.resize((image_width, image_height))
            resized_image_gray = resized_image.convert('L')
            resized_image_rgb = PILtoTorch(resized_image)
            resized_image_gray = PILtoTorch(resized_image_gray)
            self.original_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0).to(self.data_device)
            self.image_gray = resized_image_gray.clamp(0.0, 1.0).to(self.data_device)

            sky_path = image_path.replace("images", "mask")[:-10]
            sky_path = sky_path + image_path[-7:]
            if os.path.exists(sky_path):
                self.sky_mask = torch.tensor(cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)).to(self.data_device).squeeze()/255
                self.sky_mask = erode(self.sky_mask[None,None].float()).squeeze()
                self.sky_mask = torch.nn.functional.interpolate(self.sky_mask[None,None], size=(image_height,image_width), mode='bilinear', align_corners=False).squeeze()
                self.sky_mask = (self.sky_mask < 0.5).to(self.data_device)
        else:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            

        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)


        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.image_gray.cuda()
        else:
            image = Image.open(self.image_path)
            resized_image = image.resize((self.image_width, self.image_height))
            resized_image_gray = resized_image.convert('L')
            resized_image_rgb = PILtoTorch(resized_image)
            resized_image_gray = PILtoTorch(resized_image_gray)
            gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            gt_image_gray = resized_image_gray.clamp(0.0, 1.0)
            return gt_image.cuda(), gt_image_gray.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen =  w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam

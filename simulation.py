import sys
import wandb
import argparse
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf
import point_cloud_utils as pcu
from pytorch3d.loss import chamfer_distance

# Gaussian splatting dependencies
sys.path.append("gs")
from scene.gaussian_model import GaussianModel
from diff_plane_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_renderer import render, GaussianModel

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3, wp_clamp, update_param
from mpm_solver_warp.warp_utils import torch2warp_float
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
sys.path.append("utils")
from decode_param import *
from transformation_utils import *
from camera_view_utils import *
from render_utils import *
from save_video import save_video
from threestudio_utils import cleanup
from update_grad import update_grad_param, lr_scheduler

from video_distillation.cogv_guidance import CogVideoGuidance

torch.manual_seed(0)

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)

SCALE_E = 1e7

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1, material=None):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path, material)
    return gaussians


def load_inpaint_gs(model_path):
    checkpt_path = os.path.join(
        model_path, "inpaint_points.ply"
    )
    if not os.path.exists(checkpt_path):
        return None
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def render_frame(mpm_solver, gs_num, init_len, moving_pts_path, 
                current_camera, gaussians, params_inpaint,
                opacity, shs,
                unselected_pos, unselected_cov, unselected_opacity, unselected_shs):

    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
    cov3D = mpm_solver.export_particle_cov_to_torch()
    rot = mpm_solver.export_particle_R_to_torch()
    
    cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
    rot = rot.view(-1, 3, 3)[:gs_num].to(device)

    pos = pos[:init_len,:]
    pos = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(pos), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )
    cov3D = cov3D / (scale_origin * scale_origin)
    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
    if os.path.exists(moving_pts_path):
        # print("---select moving points")
        pos = torch.cat([pos, unselected_pos], dim=0)
        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
        shs = torch.cat([shs_render, unselected_shs], dim=0)
    if params_inpaint is not None:
        pos = torch.cat([pos, params_inpaint['pos']], dim=0)
        cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
        opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
        shs = torch.cat([shs, params_inpaint['shs']], dim=0)

    colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
    rendering, _, _, _ = rasterize(
        means3D=pos,
        means2D=pos,
        means2D_abs=pos,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D,
    )
    return rendering


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--downsample", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--n_key_frame", type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")

    guidance_path = os.path.join(args.model_path, 'images_generated')
    if not os.path.exists(guidance_path):
        AssertionError("Guidance frames do not exist!")

    if args.output_path is not None:
        args.output_path = args.output_path + f'_ds{args.downsample}_ep{args.n_epoch}'
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    if args.debug:
        if not os.path.exists(f"{args.output_path}/log"):
            os.makedirs(f"{args.output_path}/log")

    # Create experiment logger
    wandb.init(project="Phys3D")
    wandb.run.name = args.output_path.split('/')[-1]


    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
        optimize_params
    ) = decode_param_json(args.physics_config)

    if args.downsample != 1.:
        for k in optimize_params["line"]:
            optimize_params["line"][k] *= args.downsample
        optimize_params["bbox_2d"] = [v*args.downsample for v in optimize_params["bbox_2d"]]

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, material=material_params["material"])
    gaussians_inpaint = load_inpaint_gs(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)
    params_inpaint = None
    if gaussians_inpaint is not None:
        params_inpaint = load_params_from_gs(gaussians_inpaint, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    
    # optimize moving parts only
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )

    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        import point_cloud_utils as pcu
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        thres = 0.5 / material_params["n_grid"]
        if "playdoh" in model_path:
            thres = 1.0 / material_params["n_grid"]
        freeze_mask = find_far_points(
            init_pos, moving_pts, thres=thres
        ).bool()
        moving_pts.to("cpu")
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]


    # rorate and translate object
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)


    scaling = 1.0
    if 'cat' in model_path:
        scaling = 0.7
    if 'letter' in model_path:
        scaling = 2.0
    if 'cream' in model_path:
        scaling = 0.8
    if 'toothpaste' in model_path:
        scaling = 0.6
    if 'playdoh' in model_path:
        scaling = 0.75
    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, scaling=scaling)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            f"{args.output_path}/log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, f"{args.output_path}/log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:]), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print("gs.xyz", gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 3)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity


    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    set_boundary_conditions(mpm_solver, bc_params, time_params)
    
    tape = wp.Tape()


    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    guidance = CogVideoGuidance(guidance_path, downsample=args.downsample, num_frames=args.n_key_frame)

    current_camera, camera_view_info = get_camera_view(
        model_path,
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        default_camera_index=camera_params["default_camera_index"],
        downsample=args.downsample
    )
    rasterize = initialize_resterize(
        current_camera, gaussians, pipeline, background
    )

    ## To render the first frame as image prompt
    opacity_render = opacity
    shs_render = shs
    init_len = mpm_init_pos.shape[0]
    image_prompt = render_frame(mpm_solver, gs_num, init_len, moving_pts_path, 
                                current_camera, gaussians, params_inpaint,
                                opacity_render, shs_render,
                                unselected_pos, unselected_cov, unselected_opacity, unselected_shs)

    
    # optimization settings
    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    opt_frame_dt = time_params["opt_frame_dt"]
    step_per_frame = int(frame_dt / substep_dt)
    step_per_opt_frame = int(opt_frame_dt / substep_dt)

    stage_num = 10
    frame_per_stage = args.n_key_frame
    batch_num = args.n_epoch
    render_batch_num = 10
    optimize = not args.eval
    height = None
    width = None

    lr = {}
    for param_key in optimize_params['lr']:
        lr[param_key] = optimize_params['lr'][param_key][0]


    for batch in range(batch_num+1):
        # load ckpt
        if not optimize:
            batch = batch_num
            print('-----> loading simulation parameters')
            sim_params = torch.load(os.path.join(args.output_path, f'ckpt/sim.pth'))
            for param_key in sim_params:
                setattr(mpm_solver.mpm_model, param_key, torch2warp_float(sim_params[param_key].to(torch.float32)))
                print(param_key, wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).mean().item())

        print(f"======= Batch {batch}/{batch_num} =======")
        if optimize and batch != batch_num:
            loss_value = 0.
            loss_geo = []  # for PAC-NeRF
            img_list = []
            tape.reset()
            with tape:
                mpm_solver.finalize_mu_lam()
            
            for _ in range(step_per_opt_frame * (batch % stage_num)):
                # print("----- p2g2p")
                mpm_solver.p2g2p(None, substep_dt, device=device)
            
            for frame in tqdm(range(frame_per_stage)):
                current_camera, camera_view_info = get_camera_view(
                    model_path,
                    default_camera_index=camera_params["default_camera_index"],
                    center_view_world_space=viewpoint_center_worldspace,
                    observant_coordinates=observant_coordinates,
                    current_frame=frame,
                    downsample=args.downsample
                )
                rasterize = initialize_resterize(
                    current_camera, gaussians, pipeline, background
                )
                
                for _ in range(step_per_opt_frame * (1 + stage_num) - 1):
                    mpm_solver.p2g2p(frame, substep_dt, device=device)
                with tape:
                    mpm_solver.p2g2p(frame, substep_dt, device=device)

                    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                    cov3D = mpm_solver.export_particle_cov_to_torch()
                    rot = mpm_solver.export_particle_R_to_torch()
                
                cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:gs_num].to(device)

                pos = pos[:init_len,:]
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )


                # point cloud supervision
                pcd_path = os.path.join(args.model_path, f'pcd/{frame+1}.ply')
                if os.path.exists(pcd_path):
                    pts_gt = pcu.load_mesh_v(pcd_path)
                    pts_gt = torch.from_numpy(pts_gt).float().to("cuda")
                    loss_pts = chamfer_distance(pos[None], pts_gt[None])[0]
                    loss_geo.append(loss_pts)


                cov3D = cov3D / (scale_origin * scale_origin)
                cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                opacity = opacity_render
                shs = shs_render
                if os.path.exists(moving_pts_path):
                    # print("---select moving points")
                    pos = torch.cat([pos, unselected_pos], dim=0)
                    cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    shs = torch.cat([shs_render, unselected_shs], dim=0)
                if params_inpaint is not None:
                    pos = torch.cat([pos, params_inpaint['pos']], dim=0)
                    cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
                    opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
                    shs = torch.cat([shs, params_inpaint['shs']], dim=0)

                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                rendering, raddi, _, _ = rasterize(
                    means3D=pos,
                    means2D=pos,
                    means2D_abs=pos,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D,
                )
                img_list.append(rendering)
            
            ## Optimize
            loss = 0.
            img_list = torch.stack(img_list)

            # run guidance
            guidance_out = guidance(img_list, image_prompt.unsqueeze(0), 
                                    optimize_params["bbox_2d"], optimize_params["reduction"])

            if len(loss_geo):
                guidance_out['loss_geo'] = torch.stack(loss_geo).mean() * 10.

            for name, value in guidance_out.items():
                if name.startswith('loss_'):
                    loss += value
            loss = loss / stage_num
            if mpm_solver.mpm_model.material == 2 and optimize_params["reduction"] == "mean":
                loss *= 1e6
            if mpm_solver.mpm_model.material == 5 and optimize_params["reduction"] == "mean":
                loss *= 1e4
            print("loss:", loss.item())
            loss.backward(retain_graph=True)
            loss_value += loss.item()

            grad_x = mpm_solver.mpm_state.particle_x.grad
            grad_cov = mpm_solver.mpm_state.particle_cov.grad
            grad_r = mpm_solver.mpm_state.particle_R.grad
            loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            print(f"grad_x: max={torch.max(wp.to_torch(grad_x)).item()}, min={torch.mean(wp.to_torch(grad_x)).item()}")
            
            wp.launch(sum_vec3, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_x, grad_x], [loss_wp], device=device)
            wp.launch(sum_array, mpm_solver.n_particles*6, [mpm_solver.mpm_state.particle_cov, grad_cov], [loss_wp], device=device)
            wp.launch(sum_mat33, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_R, grad_r], [loss_wp], device=device)
            tape.backward(loss=loss_wp)

            # ================= Logging ==========================
            wandb.log({
                'loss/img': loss_value,
                'loss/wp': wp.to_torch(loss_wp).mean().item()
            })

            # E and nu
            if mpm_solver.mpm_model.material == 0:  # elastic
                update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, 1,
                                lrate=lr['E'], lower=-4.0, upper=-0.4, log_name="E", scale=mpm_solver.n_particles)
                update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, 1,
                                lrate=lr['nu'], lower=-4.0, upper=-0.4, log_name="nu", scale=mpm_solver.n_particles)
                
            if mpm_solver.mpm_model.material in [1, 4]:  # metal or plasticine
                update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, 1,
                                lrate=lr['E'], lower=-4.0, upper=-0.4 if mpm_solver.mpm_model.material==4 else 0.5, 
                                log_name="E", scale=mpm_solver.n_particles)
                update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, 1,
                                lrate=lr['nu'], lower=-4.0, upper=-0.4, log_name="nu", scale=mpm_solver.n_particles)

                update_grad_param(mpm_solver.mpm_model.yield_stress, mpm_solver.mpm_model.yield_stress.grad, 1,
                            lrate=lr['yield_stress'], lower=-4.0, upper=-1.0 if mpm_solver.mpm_model.material==4 else 0.0, 
                            log_name="yield_stress", scale=mpm_solver.n_particles)
                
            if mpm_solver.mpm_model.material == 2:  # sand
                update_grad_param(mpm_solver.mpm_model.friction_angle, mpm_solver.mpm_model.friction_angle.grad, 1, 
                                lrate=lr['friction_angle'], lower=0.0, upper=2.0, log_name="friction_angle", scale=mpm_solver.n_particles)
            
            if mpm_solver.mpm_model.material == 3:  # foam
                update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, 1,
                                lrate=lr['E'], lower=-4.0, upper=-0.4, log_name="E", scale=mpm_solver.n_particles)
                update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, 1,
                                lrate=lr['nu'], lower=-4.0, upper=-0.4, log_name="nu", scale=mpm_solver.n_particles)
                                
                update_grad_param(mpm_solver.mpm_model.yield_stress, mpm_solver.mpm_model.yield_stress.grad, 1,
                            lrate=lr['yield_stress'], lower=-4.0, upper=-0.8, log_name="yield_stress", scale=mpm_solver.n_particles)
                update_grad_param(mpm_solver.mpm_model.plastic_viscosity, mpm_solver.mpm_model.plastic_viscosity.grad, 1,
                            lrate=lr['plastic_viscosity'], lower=-4.0, upper=-1.0, log_name="plastic_viscosity", scale=mpm_solver.n_particles)
            
            if mpm_solver.mpm_model.material == 6:  # non-newtonian
                if batch < 0.5 * batch_num:
                    update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, 1,
                                    lrate=lr['E'], lower=-7.0, upper=-0.4, log_name="E", scale=mpm_solver.n_particles)
                    update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, 1,
                                    lrate=lr['nu'], lower=-4.0, upper=-0.31, log_name="nu", scale=mpm_solver.n_particles)
                else:                    
                    update_grad_param(mpm_solver.mpm_model.yield_stress, mpm_solver.mpm_model.yield_stress.grad, 1,
                                lrate=lr['yield_stress'], lower=-4.0, upper=-0.8, log_name="yield_stress", scale=mpm_solver.n_particles)
                    update_grad_param(mpm_solver.mpm_model.plastic_viscosity, mpm_solver.mpm_model.plastic_viscosity.grad, 1,
                                lrate=lr['plastic_viscosity'], lower=-4.0, upper=-1.0, log_name="plastic_viscosity", scale=mpm_solver.n_particles)
                    
                fluid_viscosity = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (2. * (1. + wp.to_torch(mpm_solver.mpm_model.nu)))
                bulk_modulus = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (3. * max(1. - 2. * wp.to_torch(mpm_solver.mpm_model.nu), 1e-4))
                print(f"   --> fluid_viscosity: {torch.mean(fluid_viscosity).item()}")
                print(f"   --> bulk_modulus: {torch.mean(bulk_modulus).item()}")

            if mpm_solver.mpm_model.material == 5:  # newtonian
                update_grad_param(mpm_solver.mpm_model.E, mpm_solver.mpm_model.E.grad, 1,
                                lrate=lr['E'], lower=-7.0, upper=-0.4, log_name="E", scale=mpm_solver.n_particles)
                update_grad_param(mpm_solver.mpm_model.nu, mpm_solver.mpm_model.nu.grad, 1,
                                lrate=lr['nu'], lower=-4.0, upper=-0.31, log_name="nu", scale=mpm_solver.n_particles)
                
                fluid_viscosity = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (2. * (1. + wp.to_torch(mpm_solver.mpm_model.nu)))
                bulk_modulus = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (3. * max(1. - 2. * wp.to_torch(mpm_solver.mpm_model.nu), 1e-4))
                print(f"   --> fluid_viscosity: {torch.mean(fluid_viscosity).item()}")
                print(f"   --> bulk_modulus: {torch.mean(bulk_modulus).item()}")
            

            for param_key in optimize_params['lr']:
                param_lr = optimize_params['lr'][param_key]
                warmup = 0 if len(param_lr) < 4 else param_lr[3]
                if mpm_solver.mpm_model.material == 6:
                    max_steps = None if len(param_lr) < 3 else param_lr[2]//2
                    if batch < 0.5 * batch_num:
                        if param_key in ['E', 'nu']:
                            lr[param_key] = lr_scheduler(optimize_params['lr'][param_key][0], optimize_params['lr'][param_key][1], 
                                    batch, batch_num//2, warmup_steps=warmup, max_steps=max_steps)
                    else:
                        if param_key in ['yield_stress', 'plastic_viscosity']:
                            lr[param_key] = lr_scheduler(optimize_params['lr'][param_key][0], optimize_params['lr'][param_key][1], 
                                    batch-batch_num//2, batch_num//2, warmup_steps=warmup, max_steps=max_steps)
                else:
                    max_steps = None if len(param_lr) < 3 else param_lr[2]
                    lr[param_key] = lr_scheduler(param_lr[0], param_lr[1], batch, batch_num, warmup_steps=warmup, max_steps=max_steps)



            # ================= Logging ==========================
            logs = {}
            for param_key in optimize_params['lr']:
                logs[f'param/{param_key}'] = wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).mean().item()
                if param_key in lr:
                    logs[f'lr/{param_key}'] = lr[param_key]
            wandb.log(logs)
                
                
            if (batch + 1) % render_batch_num == 0 or batch + 1 == batch_num:
                os.makedirs(os.path.join(args.output_path, 'ckpt'), exist_ok=True)
                print("-----> saving simulation parameters")
                param_dict = {}
                param_js = {}
                for param_key in optimize_params['lr']:
                    print(f"{param_key}:", wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).mean().item())
                    param_dict[f'{param_key}'] = wp.to_torch(getattr(mpm_solver.mpm_model, param_key))
                    param_js[f'{param_key}'] = wp.to_torch(getattr(mpm_solver.mpm_model, param_key)).mean().item()
                torch.save(param_dict, os.path.join(args.output_path, f'ckpt/sim.pth'))
                with open(os.path.join(args.output_path, f'ckpt/sim.json'), 'w') as outfile:
                    json.dump(param_js, outfile, indent=4)

            
            mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov)
            torch.cuda.empty_cache()


        # render video
        if batch == 0 or (batch + 1) % render_batch_num == 0 or batch == batch_num:
            if mpm_solver.mpm_model.material in [5, 6]:
                fluid_viscosity = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (2. * (1. + wp.to_torch(mpm_solver.mpm_model.nu)))
                bulk_modulus = SCALE_E * wp.to_torch(mpm_solver.mpm_model.E) / (3. * max(1. - 2. * wp.to_torch(mpm_solver.mpm_model.nu), 1e-4))
                print(f"   --> fluid_viscosity: {torch.mean(fluid_viscosity).item()}")
                print(f"   --> bulk_modulus: {torch.mean(bulk_modulus).item()}")

            cv2_frames = []
            renderings = []
            mpm_solver.finalize_mu_lam()
            _stage_num = int(stage_num * 1.2)
            if 'plane' in args.model_path:
                _stage_num = stage_num
            for frame in tqdm(range(_stage_num * frame_per_stage)):
                delta_r = camera_params["delta_r"]
                if batch == batch_num:
                    if 'alocasia' in args.model_path:
                        delta_r = camera_params["delta_r"] if frame < (_stage_num * frame_per_stage)/2 else camera_params["delta_r"] / frame * ((stage_num * frame_per_stage)-frame)
                        
                current_camera, camera_view_info = get_camera_view(
                    model_path,
                    default_camera_index=camera_params["default_camera_index"],
                    center_view_world_space=viewpoint_center_worldspace,
                    observant_coordinates=observant_coordinates,
                    current_frame=frame,
                    move_camera=camera_params["move_camera"] if batch == batch_num else False,
                    delta_a=camera_params["delta_a"] if batch == batch_num else None,
                    delta_e=camera_params["delta_e"] if batch == batch_num else None,
                    delta_r=delta_r,
                    downsample=args.downsample
                )
                rasterize = initialize_resterize(
                    current_camera, gaussians, pipeline, background
                )
                
                for _ in range(step_per_frame):
                    mpm_solver.p2g2p(frame, substep_dt, device=device)

                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D = mpm_solver.export_particle_cov_to_torch()
                rot = mpm_solver.export_particle_R_to_torch()
                
                cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:gs_num].to(device)

                pos = pos[:init_len,:]
                pos = apply_inverse_rotations(
                    undotransform2origin(
                        undoshift2center111(pos), scale_origin, original_mean_pos
                    ),
                    rotation_matrices,
                )
                cov3D = cov3D / (scale_origin * scale_origin)
                cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                opacity = opacity_render
                shs = shs_render
                if os.path.exists(moving_pts_path):
                    pos = torch.cat([pos, unselected_pos], dim=0)
                    cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                    opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                    shs = torch.cat([shs_render, unselected_shs], dim=0)
                if params_inpaint is not None:
                    pos = torch.cat([pos, params_inpaint['pos']], dim=0)
                    cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
                    opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
                    shs = torch.cat([shs, params_inpaint['shs']], dim=0)

                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                rendering, raddi, _, _ = rasterize(
                    means3D=pos,
                    means2D=init_screen_points,
                    means2D_abs=init_screen_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=None,
                    rotations=None,
                    cov3D_precomp=cov3D,
                )
                if (frame+1) % stage_num == 0:
                    renderings.append(rendering)

                cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                if height is None or width is None:
                    height = cv2_img.shape[0] // 2 * 2
                    width = cv2_img.shape[1] // 2 * 2
                assert args.output_path is not None
                os.makedirs(os.path.join(args.output_path, 'frames'), exist_ok=True)
                cv2.imwrite(
                    os.path.join(args.output_path, f"frames/{frame:04d}.png"),
                    255 * cv2_img,
                )
                cv_img = (np.clip(cv2_img, 0, 1)*255).astype(np.uint8)
                if optimize_params["line"]["axis"] == 0:
                    line_st = max(optimize_params["line"]["start"], 0)
                    line_ed = min(optimize_params["line"]["end"], width-1)
                    line = cv_img[optimize_params["line"]["pos"]-1:optimize_params["line"]["pos"], 
                                  optimize_params["line"]["start"]:optimize_params["line"]["end"]]
                    line_shape = line.shape[1]
                else:
                    line_st = max(optimize_params["line"]["start"], 0)
                    line_ed = min(optimize_params["line"]["end"], height-1)
                    line = cv_img[optimize_params["line"]["start"]:optimize_params["line"]["end"], 
                                  optimize_params["line"]["pos"]-1:optimize_params["line"]["pos"]]
                    line_shape = line.shape[0]
                cv2_frames.append(line)

            if batch != batch_num:
                save_video(os.path.join(args.output_path, 'frames'), os.path.join(args.output_path, 'video%02d.mp4' % batch))
            else:
                print("-----> saving final video")
                save_video(os.path.join(args.output_path, 'frames'), os.path.join(args.output_path, 'video_final.mp4'))
                sys.exit(0)
            mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov)

            cv2_frames = np.concatenate(cv2_frames, axis=0)
            cv2_frames = cv2.resize(cv2_frames, (line_shape, line_shape))[:,:,::-1]
            wandb.log({"frames": wandb.Image(cv2_frames, caption=f'renderings, ep:{batch}')})

            # save log of optical flows
            img_list = torch.stack(renderings)
            rgb_flows, flow_imgs = guidance.predict_flow(img_list, image_prompt.unsqueeze(0))

            if not os.path.exists(os.path.join(args.output_path, 'debug_flow')):
                os.makedirs(os.path.join(args.output_path, 'debug_flow'))
            for i, flow_img in enumerate(flow_imgs):
                cv2_img = flow_img.permute(1, 2, 0).detach().cpu().numpy()
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.output_path, f"debug_flow/{batch:02d}_{i:05d}_render.png"), cv2_img)

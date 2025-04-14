import torch

def apply_grid_bc_w_freeze_pts(grid_size, grid_lim, freeze_pts, mpm_solver):

    device = freeze_pts.device

    grid_pts_cnt = torch.zeros(
        (grid_size, grid_size, grid_size), dtype=torch.int32, device=device
    )

    dx = grid_lim / grid_size
    inv_dx = 1.0 / dx

    freeze_pts = (freeze_pts * inv_dx).long()

    for x, y, z in freeze_pts:
        grid_pts_cnt[x, y, z] += 1

    freeze_grid_mask = grid_pts_cnt >= 1

    freeze_grid_mask_int = freeze_grid_mask.type(torch.int32)

    number_freeze_grid = freeze_grid_mask_int.sum().item()
    print("number of freeze grid", number_freeze_grid)

    mpm_solver.enforce_grid_velocity_by_mask(freeze_grid_mask_int)

    # add debug section:

    return freeze_grid_mask
import torch

def ray_generation(img_width : int, img_height : int, K_matrix : torch.Tensor, E_matrix : torch.Tensor):
    """
    Create the directional vectors, d, from intrinsic and extrinsic matrices for a given set of image dims.
    
    Args: 
        img_width (int): image width
        img_height (int): image height 
        K_matrix (torch.Tensor): camera intrinsic matrix with shape [3, 3]
        E_matrix (torhc.Tensor): camera extrinsic matrix with shape [4, 4]
    
    Returns: 
        ray_directions (torch.Tensor): matrix with shape [img_width, img_height, 3] that includes direction, d.
        Oc (torch.Tensor): matrix with shape [1, 3] that includes the camera origin of the generated rays.
    """
    # coords generation (img_height, img_width, 3)
    y, x = torch.meshgrid(torch.arange(img_width), torch.arange(img_height), indexing='ij')
    coords = torch.cat([x, y, torch.ones_like(x)], dim=-1).float()
    coords_flat = coords.view(-1, 3)
    
    # Map pixel coordinates to normalized camera coordinates
    cam_coords_flat = torch.matmul(coords_flat, torch.linalg.inv(K_matrix).T)
    cam_coords = cam_coords_flat.view(img_height, img_width, 3)

    # Add homogeneous coordinate for transformation
    cam_coords_hom = torch.cat([cam_coords, torch.ones((img_height, img_width, 1))], dim=-1)

    # Map camera coordinates to world coordinates
    world_coords = torch.matmul(cam_coords_hom, torch.linalg.inv(E_matrix).T)

    # Extract the camera origin from E_matrix
    Oc = E_matrix[:3, 3]

    # Calculate ray directions from the camera origin to the world coordinates
    ray_directions = world_coords[..., :3] - Oc
    ray_directions = ray_directions / torch.linalg.norm(ray_directions, dim=-1, keepdim=True)

    return ray_directions, Oc

def ray_sampling(Oc: torch.Tensor, ray_direction: torch.Tensor, near_bound: int, far_bound: int, num_samples: int):
    """
    Samples points along ray directions from camera origin
    
    Args: 
        Oc (torch.Tensor): Camera Origin in shape [3, ]
        ray_direction (torch.Tensor): ray_direction in shape [img_height, img_width, 3]
        near_bound (int): start boundary for t_vals
        far_bound (int): end boundary for t_vals
        num_steps (int): number of points between near and far boundaries
    
    Returns: 
        points (torch.Tensor): points along each ray [img_height, img_width, num_samples, 3]
        t_vals (torch.Tensor): t_vals along the ray eqaution [num_samples, 1]
    """
    # Generating t_vals for r(t)
    t_vals = torch.linspace(0., 1., steps=num_samples, dtype=torch.float32)
    z_vals = near_bound * (1.-t_vals) + far_bound * (t_vals)
    
    mids = .5 * (z_vals[1:] + z_vals[:-1])
    upper = torch.cat([mids, z_vals[-1:]], dim=-1)
    lower = torch.cat([z_vals[:1], mids], dim=-1)
    t_rand = torch.rand([num_samples], device=z_vals.device)
    z_vals = lower + (upper - lower) * t_rand
    
    z_vals = z_vals.expand(list(Oc.shape[:-1]) + [num_samples])

    # Generating all r(t) for t_val range
    points = Oc[..., None, :] + ray_direction[..., None, :] * z_vals[..., :, None]
    points = points.view(-1, num_samples, 3)
    z_vals = z_vals.view(-1, num_samples)
    return points, z_vals
    
    
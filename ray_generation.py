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
    x, y = torch.meshgrid(torch.arange(img_width), torch.arange(img_height))
    coords = torch.cat([x, y, torch.ones_like(x)], dim=-1)
    coords = coords.view((img_height, img_width, 3)).float()
    
    # coords (img_height, img_width, 3) x K_matrix (3, 3)
    cam_coords = torch.matmul(coords, torch.linalg.inv(K_matrix))
    cam_coords = torch.cat([cam_coords, torch.ones((img_height, img_width, 1))], dim=-1)
    cam_coords = cam_coords.view((img_height, img_width, 4))
    
    # Cam_coords (img_height, img_width, 4) x E_matrix (4, 4)
    world_coords = torch.matmul(cam_coords, torch.linalg.inv(E_matrix))
    
    # world_coords - Camera_Origin / |world_coords - Camera_Origin|_2
    Oc = E_matrix[:3, 3]
    ray_directions = world_coords[..., :3] - Oc
    ray_directions = ray_directions / torch.linalg.norm(ray_directions, dim=-1, keepdim=True)  # Normalize

    return ray_directions, Oc

def ray_sampling(Oc: torch.Tensor, ray_direction: torch.Tensor, near_bound: int, far_bound: int, num_steps: int):
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
    t_vals = torch.linspace(near_bound, far_bound, steps=num_steps, dtype=torch.float32)
    
    # Generating all r(t) for t_val range
    points = Oc.unsqueeze(0).unsqueeze(0) + ray_direction.unsqueeze(-2) * t_vals.unsqueeze(1)
    return points, t_vals
    
    
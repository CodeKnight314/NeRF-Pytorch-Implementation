import torch

def getImageCoordinates(img_h: int, img_w: int):
    """
    Generating 2D image plane of all pixel coordinates
    
    Args: 
        img_h (int): Height of the image
        img_w (int): Width of the image
    
    Return: 
        grid (torch.Tensor): Grid of x, y values with grid shaped [3, H, W]
    """
    x = torch.arange(img_w)
    y = torch.arange(img_h)

    x, y = torch.meshgrid(x, y, indexing='ij')
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    grid = torch.cat([x, y, torch.ones((img_h, img_w, 1))], dim=-1)
    return grid

def generateWorldCoordinates(grid: torch.Tensor, K: torch.Tensor):
    """
    Generating direction vectors from the image plane
    
    Args: 
        grid (torch.Tensor): Grid of x, y values with grid shaped [3, H, W]
        K (torch.Tensor): the camera intrinsic matrix with shape [3, 3]
    
    Returns: 
        world_coord (torch.Tensor): projected image coordinates to world coordinates with shape [3, H*W]
    """
    world_coord = torch.linalg.inv(K) @ grid.reshape(3, -1)
    return world_coord

def generateDirection(world_coord: torch.Tensor, E: torch.Tensor):
    """
    Generating direction vectors from the image plane
    
    Args: 
        world_coord (torch.Tensor): projected image coordinates to world coordinates with shape [3, H*W]
        E (torch.Tensor): the camera extrinsic matrix with shape [4, 4]
    
    Returns: 
        direction (torch.Tensor): Normalized Direction Tensor that ha shape [3, H*W]
        t (torch.Tensor): Origin Tensor that has shape [1, 3]
    """
    R = E[:3, :3]
    t = E[:3, 3]

    direction = R @ world_coord
    direction = direction / torch.norm(direction, dim=0, keepdim=True)
    return direction, t

def ray_generation(img_h: int, img_w: int, K: torch.Tensor, E: torch.Tensor):
    """
    Generating rays from the image plane
    
    Args: 
        img_h (int): the height of the image
        img_w (int): the width of the image 
        K (torch.Tensor): the camera intrinsic matrix with shape [3, 3]
        E (torch.Tensor): the camera extrinsic matrix with shape [4, 4]
    
    Returns:
        direction (torch.Tensor): Normalized Direction Tensor that ha shape [3, H*W]
        origin (torch.Tensor): Origin Tensor that has shape [1, 3]
    """
    grid = getImageCoordinates(img_h, img_w)
    world_coord = generateWorldCoordinates(grid, K)
    direction, origin = generateDirection(world_coord, E)
    return direction, origin

def ray_sampling(origin: torch.Tensor, direction: torch.Tensor, near: float, far: float, num_steps: int, mode: str = "uniform"):
    """
    Sample points along rays from origin in the direction.
    
    Args: 
        origin (torch.Tensor): Origin Tensor that has shape [1, 3]
        direction (torch.Tensor): Normalized Direction Tensor that ha shape [3, H*W]
        near (float): The lower limit for ray marching
        far (float): The upper limit for ray marching
        num_steps (int): The number of samples between the lower and upper bound
        mode (str): The mode of sampling for the points
        
    Returns: 
        points (torch.Tensor): points with shape [3, num_steps, H*W]
        t_vals (torch.Tensor): t_vals with shape [100, 1]
    """
    if mode == "stratified":
        offset = torch.rand((1, num_steps), device=direction.device)
        t_vals = torch.linspace(near, far, num_steps, device=direction.device)
        t_vals = t_vals + offset * ((far - near) / num_steps)
    else:
        t_vals = torch.linspace(near, far, num_steps, device=direction.device)

    points = direction[:, None, :] * t_vals[..., None] + origin[:, None, None]

    return points, t_vals

def sample_fine_points(coarse_points: torch.Tensor, coarse_densities: torch.Tensor, fine_samples: int) -> torch.Tensor:
    """
    Samples additional fine points around regions of interest based on the densities from the coarse model.
    Supports batched input.

    Args:
        coarse_points (torch.Tensor): Coarse sampled points of shape (batch_size, num_rays, num_steps, 3).
        coarse_densities (torch.Tensor): Densities predicted by the coarse model of shape (batch_size, num_rays, num_steps).
        fine_samples (int): Number of fine samples to generate per ray.

    Returns:
        fine_points (torch.Tensor): Fine sampled points of shape (batch_size, num_rays, fine_samples, 3).
    """
    batch_size, num_rays, num_steps, _ = coarse_points.shape

    weights = coarse_densities + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    u = torch.rand(batch_size, num_rays, fine_samples, device=coarse_points.device)

    cdf_flattened = cdf.reshape(batch_size * num_rays, -1)
    u_flattened = u.reshape(batch_size * num_rays, fine_samples)

    fine_indices = torch.searchsorted(cdf_flattened, u_flattened, right=True) - 1
    fine_indices = fine_indices.clamp(0, cdf.shape[-1] - 2)

    fine_indices = fine_indices.view(batch_size, num_rays, fine_samples)

    low = fine_indices
    high = fine_indices + 1

    low_expanded = low.unsqueeze(-1)
    high_expanded = high.unsqueeze(-1)

    low_cdf = torch.gather(cdf, -1, low_expanded).squeeze(-1)
    high_cdf = torch.gather(cdf, -1, high_expanded).squeeze(-1)

    low_points = torch.gather(coarse_points, 2, low_expanded.expand(-1, -1, -1, 3))
    high_points = torch.gather(coarse_points, 2, high_expanded.expand(-1, -1, -1, 3))

    t = (u - low_cdf) / (high_cdf - low_cdf + 1e-5)
    fine_points = low_points + t.unsqueeze(-1) * (high_points - low_points)
    fine_points = fine_points.view(fine_points.shape[0], -1, 3)
    return fine_points


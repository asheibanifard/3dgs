from matplotlib import pyplot as plt
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from data_reader import DatasetReader
from scene import cameras
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, strip_symmetric
from scene.dataset_readers import CameraInfo_Xray, ConeGeometry, Xray_readNerfSyntheticInfo, angle2pose
from utils.graphics_utils import focal2fov
class GaussianModel_Xray:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    def __init__(self, d):
        self._scaling = d['_scaling']
        self._rotation = d['_rotation']
        self._xyz = d['_xyz']
        self._features_dc = d['_features_dc']
        self._features_rest = d['_features_rest']
        self._opacity = d['_opacity']
        self.active_sh_degree = d['active_sh_degree']
        self.setup_functions()
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)


def render(viewpoint_camera, pc : GaussianModel_Xray, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points    
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    compute_cov3D_python = False
    if compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = pc.get_features

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image.mean(dim=0, keepdim=True),
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,           
            "radii": radii}
import numpy as np
import numpy as np
import torch
def C2W(theta, phi, radius):
    """
    Compute the camera-to-world transformation matrix from spherical angles.
    
    theta: Elevation angle in radians
    phi: Azimuth angle in radians
    radius: Distance from the origin
    """
    # Camera position in world space
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    pos = np.array([x, y, z])

    # Forward vector (view direction)
    f = pos / np.linalg.norm(pos)  # Normalize

    # Right vector (perpendicular in x-y plane)
    r = np.array([-np.sin(phi), np.cos(phi), 0])

    # Up vector (cross product of right and forward vectors)
    u = np.cross(r, f)

    # Construct the rotation part of the transformation matrix
    c2w = np.eye(4)
    c2w[0, :3] = r
    c2w[1, :3] = u
    c2w[2, :3] = f

    # Translation part
    c2w[:3, 3] = pos

    return c2w
# # Transformation Matrices (Assuming 3x3 Rotation and 3x1 Translation)
# R = np.eye(3, dtype=np.float32)  # Identity matrix for rotation (3x3)
# T = np.zeros((3, 1), dtype=np.float32).squeeze()   # Zero translation vector (3x1)
# print(T.shape)
# # Field of View (FoV) in Degrees
# FoVx = 90.0  # Example FoV for X-axis
# FoVy = 90.0  # Example FoV for Y-axis
# # Image Data (Placeholder as a Torch Tensor)
type = 'val'
dataset = DatasetReader(file_path='data/head_50.pickle').read()
geometry = ConeGeometry(dataset)
projs = dataset[type]["projections"]
angles = dataset[type]["angles"]
h, w = projs[0].shape
fovx = focal2fov(geometry.DSD, w)
data = torch.load('checkpoint/checkpoints.pth')
d = data['gaussians']
pc = GaussianModel_Xray(d)
cam_infos = []
for idx, image_arr in enumerate(projs):
    c2w = C2W(angles[idx], torch.pi, 256*0.8)
    image_name = str(idx)

    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3]) 
    T = w2c[:3, 3]


    image = torch.tensor(image_arr).cuda()
    angle = angles[idx]

    fovy = focal2fov(geometry.DSD, h)
    FovY = fovy 
    FovX = fovx
    viewpoint_camera = cameras.Camera(R, T, FovX, FovY, image, image_name, angle)


    dpkg = render(viewpoint_camera, pc, None, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))

    plt.imshow(dpkg['render'].cpu().detach().numpy().squeeze(), cmap='gray')
    plt.savefig(f'./output/rendered_test_{idx}.png')
    # cam_infos.append(CameraInfo_Xray(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_name=image_name, width=image.shape[0], height=image.shape[1], angle=angle))
        
# print(cam_infos[0])
# image = dataset['image']
# image = torch.tensor(image, dtype=torch.float32, device="cuda")
# # Ground Truth Alpha Mask (Binary Mask for Transparency)

# # Image Name (String)
# image_name = "default_image.png"

# # Angle (Assuming in Radians)
# angle = 0.06283185  # Starting angle at 0 radians

# # viewpoint_camera = cameras.Camera(R, T, FoVx, FoVy, image, image_name, angle)
# pc = GaussianModel_Xray(d)

# dpkg = render(viewpoint_camera, pc, None, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))

# plt.imshow(dpkg['render'].cpu().detach().numpy().squeeze())
# plt.savefig('./output/rendered_test.png')
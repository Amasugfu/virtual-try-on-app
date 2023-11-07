import torch


def partial_mesh(model):
    pass


def poisson_surface_reconsturction():
    pass


def refine_geometry(point_cloud):
    pass


def map_texture():
    pass


class Camera:
    def __init__(self, fx=1, fy=1, centerx=None, centery=None) -> None:
        self.fx = fx
        self.fy = fy
        self.centerx = centerx
        self.centery = centery

    def project(self, pm_depth: torch.Tensor):
        """
        @param: pm_depth: batch x layers x 1 x H x W
        """
        centerx = pm_depth.shape[-1]/2 if self.centerx is None else self.centerx
        centery = pm_depth.shape[-2]/2 if self.centery is None else self.centery

        normx = torch.arange(pm_depth.shape[-1]).reshape(1, 1, -1) - centerx
        normy = torch.arange(pm_depth.shape[-2]).reshape(1, -1, 1) - centery

        x = pm_depth*normx/self.fx
        y = pm_depth*normy/self.fy

        point_cloud = torch.concat([x, y, pm_depth], dim=-3)
        return point_cloud


class GarmentModel3D:
    def __init__(self,
                 pm_depth: torch.Tensor, 
                #  pm_seg: torch.Tensor, 
                 pm_norm: torch.Tensor, 
                 pm_rgb: torch.Tensor,
                 camera: Camera = Camera(),
                 ) -> None:
        self.pm_depth = pm_depth
        # self.pm_seg = pm_seg
        self.pm_norm = pm_norm
        self.pm_rgb = pm_rgb
        self.camera = camera

    def as_obj(self):
        """
        parse to an OBJ 3D model.
        """
        pass

    def as_img(self):
        """
        parse each peelmaps to PNG images.
        """
        pass

    def reconstruct(self):
        """
        reconstruct the model as a 3d meshes using the peelmaps.
        texture is also generated.

        @return: `point_cloud: torch.Tensor`, `texture: torch.Tensor`
        """
        point_cloud = self.camera.project(self.pm_depth)
        meshes = refine_geometry(point_cloud)
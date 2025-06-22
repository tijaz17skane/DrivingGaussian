#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def eval_sh(deg, sh, dirs):
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def get_cartesian_coords(r, elev, azim, in_degrees=False):
    factor = 1
    if in_degrees:
        factor *= np.pi / 180.
    X = torch.stack((
        torch.cos(factor * elev) * torch.sin(factor * azim),
        torch.sin(factor * elev),
        torch.cos(factor * elev) * torch.cos(factor * azim)
    ), dim=2)

    return r * X.view(-1, 3)

def get_spherical_coords(X):
    r_x = torch.linalg.norm(X, dim=1)

    elev_x = torch.asin(X[:, 1] / r_x)
    elev_x[X[:, 1] / r_x <= -1] = -np.pi / 2
    elev_x[X[:, 1] / r_x >= 1] = np.pi / 2

    azim_x = torch.acos(X[:, 2] / (r_x * torch.cos(elev_x)))
    azim_x[X[:, 2] / (r_x * torch.cos(elev_x)) <= -1] = np.pi
    azim_x[X[:, 2] / (r_x * torch.cos(elev_x)) >= 1] = 0.
    azim_x[X[:, 0] < 0] *= -1

    return r_x, elev_x, azim_x

def get_samples_on_sphere(device, pole_samples=False, n_elev=10, n_azim=2*10):
    n_camera = n_elev * n_azim
    if pole_samples:
        n_camera += 2

    candidate_dist = torch.Tensor([1. for i in range(n_camera)]).to(device)

    candidate_elev = [-90. + (i + 1) / (n_elev + 1) * 180.
                      for i in range(n_elev)
                      for j in range(n_azim)]

    candidate_azim = [360. * j / n_azim
                      for i in range(n_elev)
                      for j in range(n_azim)]

    if pole_samples:
        candidate_elev = [-89.9] + candidate_elev + [89.9]
        candidate_azim = [0.] + candidate_azim + [0.]

    candidate_elev = torch.Tensor(candidate_elev).to(device).view(-1, 1)
    candidate_azim = torch.Tensor(candidate_azim).to(device).view(-1, 1)

    X_cam = get_cartesian_coords(r=candidate_dist.view(-1, 1),
                                 elev=candidate_elev,
                                 azim=candidate_azim,
                                 in_degrees=True)

    return X_cam, candidate_dist, candidate_elev, candidate_azim

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

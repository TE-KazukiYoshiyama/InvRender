from email.mime import base
import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from model.sg_render import compute_envmap
import imageio
import mcubes
import trimesh


tonemap_img = lambda x: np.power(x, 1./2.2)
clip_img = lambda x: np.clip(x, 0., 1.)

"""
[start] Most parts came from https://github.com/Totoro97/NeuS
"""
def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    Mx = len(X)
    My = len(Y)
    Mz = len(Z)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    print(f"{xi} / {Mx}, {yi} / {My}, {zi} / {Mz}")
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    pts = pts.cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max
    b_min_np = bound_min

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np) + b_min_np
    return vertices, triangles
"""
[end] Most parts came from https://github.com/Totoro97/NeuS
"""


def extract_mesh(model, meshdir, grid_size, timestamp, albedo_ratio=None):
    # Extract geometry
    bound_min = np.array([-1.01, -1.01, -1.01])
    bound_max = np.array([ 1.01,  1.01,  1.01])
    implicit_network = model.implicit_network
    sdf = lambda x: -implicit_network(x)[:, 0]
    verts, faces = extract_geometry(bound_min, bound_max, grid_size, 0.0, sdf)
    verts = verts.astype(np.float32)
    mesh = trimesh.Trimesh(verts, faces)

    # Extract vertex attributes
    n = len(verts)
    def extract_vertex_attrs(network, fname, attr_name=None, idx=0):
        vertex_attrs = np.zeros_like(verts)
        batch_size = 512
        for i in range(0, n, batch_size):
            v = verts[i:i+batch_size, :]
            v = torch.from_numpy(v).cuda()
            with torch.no_grad():
                out = network(v)
                if isinstance(out, dict):
                    vertex_attr = out[attr_name]
                else:
                    vertex_attr = out
                vertex_attr = vertex_attr.detach().cpu().numpy()
                if vertex_attr.shape[-1] != 3:
                    vertex_attr0 = np.zeros((len(vertex_attr), 3))
                    vertex_attr0[:, idx:idx+1] = vertex_attr
                    vertex_attr = vertex_attr0
                vertex_attrs[i:i+batch_size, :] = vertex_attr

        vertex_attrs = np.clip(vertex_attrs * 255.0, 0.0, 255.0).astype(np.uint8)
        mesh.visual.vertex_colors = vertex_attrs
        mesh.export(os.path.join(meshdir, '{}_G{}_{}.obj'.format(timestamp, grid_size, fname)))

    extract_vertex_attrs(model.envmap_material_network, 'base_color', 'sg_diffuse_albedo')
    extract_vertex_attrs(model.envmap_material_network, 'roughness', 'sg_roughness', 1)
    # extract_vertex_attrs(model.indirect_illum_network, 'indirect_illumination')

    print('Done extracting', meshdir)


def extract(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    mesh_folder_name = kwargs['mesh_folder_name']

    expname = 'Mat-' + f"scan{kwargs['scan_id']}"

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', mesh_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    meshdir = os.path.join('../', mesh_folder_name, expname)
    utils.mkdir_ifnotexists(meshdir)

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()
    
    # load trained model
    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    print('Loading checkpoint: ', ckpt_path)
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])

    print("start extracting...")
    model.eval()

    grid_size = kwargs["grid_size"]
    
    extract_mesh(model, meshdir, grid_size, timestamp, albedo_ratio=None)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--scan_id', type=int, default=69, help='scan id')

    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')

    parser.add_argument('--grid_size', type=int, default=512, help='Grid size')

    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    opt = parser.parse_args()

    gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    extract(conf=opt.conf,
                mesh_folder_name='meshes',
                scan_id=opt.scan_id,
                exps_folder_name=opt.exps_folder,
                grid_size=opt.grid_size, 
                timestamp=opt.timestamp,
                checkpoint=opt.checkpoint,
                )

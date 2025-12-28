import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, asin, stack
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

def normalize(a):
    normalized = a / norm(a, ord=2, axis=0, keepdims=True)
    return normalized

def cartesian_to_spherical(r_cartesian):
    """
    Convert from Cartesian coordinate to spherical coordinate
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    d = norm(r_cartesian, ord=2, axis=0)
    lat = asin(r_cartesian[2] / d)
    lon = atan2(r_cartesian[1], r_cartesian[0])
    return stack((d, lon, lat), axis=0)

def spherical_to_cartesian(r_sphere):
    """
    Convert from spherical coordinate to lon lat
    The 0th axis is (r, lon, lat), the 1st axis is the points
    """
    x = r_sphere[0] * cos(r_sphere[2]) * cos(r_sphere[1])
    y = r_sphere[0] * cos(r_sphere[2]) * sin(r_sphere[1])
    z = r_sphere[0] * sin(r_sphere[2])
    return stack((x, y, z), axis=0)

def rotate_along_z_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0]
    y = r_cartesian[1]
    z = r_cartesian[2]
    new_x = cos_angle * x - sin_angle * y
    new_y = sin_angle * x + cos_angle * y
    return stack((new_x, new_y, z), axis=0)
    
def rotate_along_x_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0]
    y = r_cartesian[1]
    z = r_cartesian[2]
    new_y = cos_angle * y - sin_angle * z
    new_z = sin_angle * y + cos_angle * z
    return stack((x, new_y, new_z), axis=0)
 
def rotate_along_y_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0]
    y = r_cartesian[1]
    z = r_cartesian[2]
    new_x = cos_angle * x - sin_angle * z
    new_z = sin_angle * x + cos_angle * z
    return stack((new_x, y, new_z), axis=0)

def compute_solid_angle(r_corners_spherical):

    r_corners_cartesian = spherical_to_cartesian(r_corners_spherical)
    vec1 = (  ( r_corners_cartesian[:, 1, :, :] - r_corners_cartesian[:, 0, :, :] ) 
            + ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 3, :, :] ) ) / 2 
    vec2 = (  ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 1, :, :] ) 
            + ( r_corners_cartesian[:, 3, :, :] - r_corners_cartesian[:, 0, :, :] ) ) / 2 

    # outer product in-place
    d0 =   vec1[1] * vec2[2] - vec1[2] * vec2[1]
    d1 = - vec1[0] * vec2[2] + vec1[2] * vec2[0]
    d2 =   vec1[0] * vec2[1] - vec1[1] * vec2[0]
    areas = (d0**2 + d1**2 + d2**2)**0.5
    solid_angles = areas / np.sum(areas) * np.pi * 4

    return solid_angles


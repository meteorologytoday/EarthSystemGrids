import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, acos, stack

#from jax.numpy.linalg import norm
#import jax.numpy as jnp
#from jax.lax import atan2, acos

def normalize(a):
    normalized = a / norm(a, ord=2, axis=0, keepdims=True)
    return normalized

def cartesian_to_spherical(r_cartesian):
    """
    Convert from Cartesian coordinate to spherical coordinate
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    d = norm(r_cartesian, ord=2, axis=0)
    lat = acos(r_cartesian[2, :] / d)
    lon = atan2(r_cartesian[0, :], r_cartesian[1, :])
    return stack((d, lon, lat), axis=0)

def spherical_to_cartesian(r_sphere, r=1.0):
    """
    Convert from spherical coordinate to lonlat
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    x = r_sphere[0, :] * cos(r_sphere[2, :]) * cos(r_sphere[1, :])
    y = r_sphere[0, :] * cos(r_sphere[2, :]) * sin(r_sphere[1, :])
    z = r_sphere[0, :] * sin(r_sphere[2, :])
    return stack((x, y, z), axis=0)

def rotate_along_z_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0, :]
    y = r_cartesian[1, :]
    z = r_cartesian[2, :]
    new_x = cos_angle * x - sin_angle * y
    new_y = sin_angle * x + cos_angle * y
    return stack((new_x, new_y, z), axis=0)
    
def rotate_along_x_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0, :]
    y = r_cartesian[1, :]
    z = r_cartesian[2, :]
    new_y = cos_angle * y - sin_angle * z
    new_z = sin_angle * y + cos_angle * z
    return stack((x, new_y, new_z), axis=0)
 
def rotate_along_y_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0, :]
    y = r_cartesian[1, :]
    z = r_cartesian[2, :]
    new_x = cos_angle * x - sin_angle * z
    new_z = sin_angle * x + cos_angle * z
    return stack((new_x, y, new_z), axis=0)
 
def generate_cubic_sphere(n: int, normalize_flag:bool=True):
    """
        normalize_flag: For debug purpose. If set to False, then the returned points are
                        not normalized. The points will form a cubic instead of a sphere.
    """
        
    # four corner points of the top face
    p1 = np.atleast_2d(np.array([ 1.0, -1.0, -1.0])).T
    p2 = np.atleast_2d(np.array([ 1.0,  1.0, -1.0])).T
    p3 = np.atleast_2d(np.array([ 1.0,  1.0,  1.0])).T
    p4 = np.atleast_2d(np.array([ 1.0, -1.0,  1.0])).T

    unit_x = normalize(p2 - p1)
    unit_y = normalize(p4 - p1)
   
    dx = unit_x * norm(p2 - p1) / n
    dy = unit_y * norm(p4 - p1) / n
 
    r_cartesian_T = np.zeros((3, n, n))
    origin = (p1 + p2 + p3 + p4) / 4 - ( dx + dy ) * n/2
    for i in range(n):
        for j in range(n):
            r_cartesian_T[:, i, j] = (origin + (0.5 + i) * dx + (0.5 + j) * dy ).flatten()
   
    if normalize_flag:
        r_cartesian_T = normalize(r_cartesian_T.reshape((3, n*n)))

    r_spherical_T = cartesian_to_spherical(r_cartesian_T)

    all_tiles = []
    for i in range(4):
        all_tiles.append(rotate_along_z_axis(r_cartesian_T, angle_rad=i*np.pi/2.0))

    all_tiles.append(rotate_along_y_axis(r_cartesian_T, angle_rad=np.pi/2.0))
    all_tiles.append(rotate_along_y_axis(r_cartesian_T, angle_rad=-np.pi/2.0))

    return all_tiles

if __name__ == "__main__":
   
    r_cartesian_T = generate_cubic_sphere(n=5)
    normalized_r_cartesian_T = [ normalize(tile) for tile in r_cartesian_T ]
     
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(azim=-30, elev=45, roll=0) 

    ax.scatter(0, 0, 0, color="red", s=10)

    for i in range(6):

        tile_r_cartesian_T = r_cartesian_T[i]
        tile_normalized_r_cartesian_T = normalized_r_cartesian_T[i]
        ax.scatter(tile_r_cartesian_T[0, :], tile_r_cartesian_T[1, :], tile_r_cartesian_T[2, :])
        ax.scatter(tile_normalized_r_cartesian_T[0, :], tile_normalized_r_cartesian_T[1, :], tile_normalized_r_cartesian_T[2, :])

#    for i in range(r_cartesian_T.shape[1]):
#        AB = stack((r_cartesian_T[:, i], normalized_r_cartesian_T[:, i]), axis=0)
#        ax.plot(AB[:, 0], AB[:, 1], AB[:, 2], color="black")

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")

    lim = np.array([-1, 1])*1.5
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    plt.show()
    
    

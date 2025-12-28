import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, asin, stack
from dataclasses import dataclass
from typing import List, Dict
from global_land_mask import globe
from pathlib import Path

@dataclass
class GenericLatLonCap:
    number_of_mid_latitude_grids: int
    number_of_longitude_grids_per_face: int
    r_spherical: List[np.ndarray]
    r_corners_spherical: List[np.ndarray]
    angle_between_grid_north_and_spherical_north: List[np.ndarray]
    grid_numbering: List[np.ndarray]
    neighbor_grid_numbering: List[np.ndarray]
    number_of_tiles: int
    binary_mask: List[np.ndarray]
    grid_solid_angles: List[np.ndarray]
    #grid_edge_lengths: List[np.ndarray] # [ tile (j, i, [ENWS]) ]
    #grid_center_lengths: List[np.ndarray] # [ tile (j, i, [ENWS]) ]

 
    # An array of shape (2 + 4*tiles_per_mid_latitude, 4, 2)
    # axis=1: 4 => [starting tile's East, North, West, South]
    # axis=2: 2 => (linked tile number, number of rot90 operations to align)
    neighbor_tile_and_rotation: np.ndarray # An array of shape () = (neighbor tile number [starting tile's East, North, West, South], number of rot90 to align)

@dataclass
class TiledLatLonCap:
    shapes: Dict[str, tuple]
    number_of_grids_per_side: int
    r_spherical: np.ndarray
    r_corners_spherical: np.ndarray
    angle_between_grid_north_and_spherical_north: np.ndarray
    grid_numbering: np.ndarray
    neighbor_grid_numbering: np.ndarray
    number_of_tiles: int
    binary_mask: np.ndarray
    neighbor_tile_and_rotation: np.ndarray # An array of shape () = (neighbor tile number [starting tile's East, North, West, South], number of rot90 to align)
    number_of_tiles_in_mid_latitude: int
    grid_solid_angles: np.ndarray

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

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

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

def generate_neighbor_tile_and_rotation(number_of_tiles_in_mid_latitude:int):
    if (not number_of_tiles_in_mid_latitude > 0) or number_of_tiles_in_mid_latitude % 1 != 0:
        raise ValueError("Error: number of tiles in mid-latitude should be a positive integer.")

    total_number_of_tiles = 4 * number_of_tiles_in_mid_latitude + 2
    number_of_side_tiles = 4 * number_of_tiles_in_mid_latitude
    north_cap_tile_number = 4 * number_of_tiles_in_mid_latitude
    south_cap_tile_number = 4 * number_of_tiles_in_mid_latitude + 1

    # dimension = ( tile, east-north-west-south tile, [neighbor_tile_number, number of rotation90 operations to align])
    unassigned_number = -9999
    neighbor_tile_and_rotation = np.zeros((total_number_of_tiles, 4, 2), dtype=int) + unassigned_number

    # Define some named index for clarity
    EAST, NORTH, WEST, SOUTH = 0, 1, 2, 3

    # ==== Side tiles: all tiles except for north and south caps ====
    side_tiles_numbering = np.arange(number_of_side_tiles, dtype=int).reshape((4, number_of_tiles_in_mid_latitude)).transpose()
    # East
    neighbor_tile_and_rotation[:number_of_side_tiles, EAST, 0] = np.roll(side_tiles_numbering, -1, axis=1).flatten()
    neighbor_tile_and_rotation[:number_of_side_tiles, EAST, 1] = 0

    # West
    neighbor_tile_and_rotation[:number_of_side_tiles, WEST, 0] = np.roll(side_tiles_numbering,  1, axis=1).flatten()
    neighbor_tile_and_rotation[:number_of_side_tiles, WEST, 1] = 0

    # Naive North and south
    neighbor_tile_and_rotation[:number_of_side_tiles, NORTH, 0] = np.roll(side_tiles_numbering, -1, axis=0).flatten()
    neighbor_tile_and_rotation[:number_of_side_tiles, SOUTH, 0] = np.roll(side_tiles_numbering,  1, axis=0).flatten()
    neighbor_tile_and_rotation[:number_of_side_tiles, NORTH, 1] = 0
    neighbor_tile_and_rotation[:number_of_side_tiles, SOUTH, 1] = 0

    # Correct the North of the top-most tiles
    neighbor_tile_and_rotation[number_of_tiles_in_mid_latitude-1::number_of_tiles_in_mid_latitude, NORTH, 0] = north_cap_tile_number
    neighbor_tile_and_rotation[1*number_of_tiles_in_mid_latitude-1, NORTH, 1] = -1 # 90 deg clockwise
    neighbor_tile_and_rotation[2*number_of_tiles_in_mid_latitude-1, NORTH, 1] =  2 # 180 deg
    neighbor_tile_and_rotation[3*number_of_tiles_in_mid_latitude-1, NORTH, 1] =  1 # 90 deg counterclockwise
    neighbor_tile_and_rotation[4*number_of_tiles_in_mid_latitude-1, NORTH, 1] =  0 # no rotation (already algiend)
    
    # Correct the South of the bottom-most tiles
    neighbor_tile_and_rotation[0::number_of_tiles_in_mid_latitude, SOUTH, 0] = south_cap_tile_number
    neighbor_tile_and_rotation[0*number_of_tiles_in_mid_latitude, SOUTH, 1] = -1 # 90 deg clockwise
    neighbor_tile_and_rotation[1*number_of_tiles_in_mid_latitude, SOUTH, 1] =  2 # 180 deg
    neighbor_tile_and_rotation[2*number_of_tiles_in_mid_latitude, SOUTH, 1] =  1 # 90 deg counterclockwise
    neighbor_tile_and_rotation[3*number_of_tiles_in_mid_latitude, SOUTH, 1] =  0 # no rotation (already algiend)
   
    # ==== Caps ==== 
    # North Cap
    neighbor_tile_and_rotation[north_cap_tile_number, (EAST, NORTH, WEST, SOUTH), 0] = (np.arange(4)+1)*number_of_tiles_in_mid_latitude - 1

    neighbor_tile_and_rotation[north_cap_tile_number, EAST,  1] =  1
    neighbor_tile_and_rotation[north_cap_tile_number, NORTH, 1] =  2
    neighbor_tile_and_rotation[north_cap_tile_number, WEST,  1] = -1
    neighbor_tile_and_rotation[north_cap_tile_number, SOUTH, 1] =  0
   
    # South Cap (rotation of north cap along x-axis by 180 degree) 
    neighbor_tile_and_rotation[south_cap_tile_number, EAST,  0] = 0*number_of_tiles_in_mid_latitude
    neighbor_tile_and_rotation[south_cap_tile_number, NORTH, 0] = 3*number_of_tiles_in_mid_latitude
    neighbor_tile_and_rotation[south_cap_tile_number, WEST,  0] = 2*number_of_tiles_in_mid_latitude
    neighbor_tile_and_rotation[south_cap_tile_number, SOUTH, 0] = 1*number_of_tiles_in_mid_latitude
    
    neighbor_tile_and_rotation[south_cap_tile_number, EAST,  1] =  -1
    neighbor_tile_and_rotation[south_cap_tile_number, NORTH, 1] =   0
    neighbor_tile_and_rotation[south_cap_tile_number, WEST,  1] =   1
    neighbor_tile_and_rotation[south_cap_tile_number, SOUTH, 1] =   2

    if np.any(neighbor_tile_and_rotation == unassigned_number):
        print("Warning: Some values of `neighbor_tile_and_rotation` are not assigned.")

    return neighbor_tile_and_rotation
 
def generate_generic_latloncap(
    lat_north_critical_degree: float,
    lat_south_critical_degree: float,
    number_of_mid_latitude_grids: int,
    number_of_longitude_grids_per_face: int,
):
    
    """
        lat_critical_degree : The latitude (in degree) beyond which polar cap starts.
        number_of_mid_latitude_grids : number of latitude boxes between [-lat_critical, lat_critical]
        number_of_longitude_grids_per_face : number of longtitude dividing one side face. Therefore,
                                             the longitude resolution = \pi / 2 / number_of_longitude_grids_per_face
    """
   
    if not ( -90.0 < lat_south_critical_degree and lat_south_critical_degree < 0 and 0 < lat_north_critical_degree and lat_north_critical_degree < 90.0 ):
        raise ValueError(f"Error: The order has to be: `-90 < lat_south_critical_degree < 0 < lat_north_critical_degree < 90`.")
 
    lat_north_critical = np.pi / 180.0 * lat_north_critical_degree
    lat_south_critical = np.pi / 180.0 * lat_south_critical_degree
    
    # mid latitude [-lat_critical, lat_critical]
    dlon = np.pi / 2 / number_of_longitude_grids_per_face
    dlat = (lat_north_critical - lat_south_critical) / number_of_mid_latitude_grids
   
    lat_bounds = np.linspace(lat_south_critical, lat_north_critical, number_of_mid_latitude_grids+1)
    lat_centers = ( lat_bounds[:-1] + lat_bounds[1:] ) / 2.0

    _stack_r_spherical = []
    _stack_r_corners_spherical = []

    # One side latlon face
    for tile_index in range(1):
        lon_bounds = (tile_index * np.pi / 2 - np.pi / 4) + np.linspace(0, np.pi/2, number_of_longitude_grids_per_face+1)
        lon_centers = ( lon_bounds[:-1] + lon_bounds[1:] ) / 2.0
        llat, llon = np.meshgrid(lat_centers, lon_centers, indexing="ij")
        
        _r_spherical = np.zeros((3, number_of_mid_latitude_grids, number_of_longitude_grids_per_face))
        _r_corners_spherical = np.zeros((3, 4, number_of_mid_latitude_grids, number_of_longitude_grids_per_face))
        for i in range(number_of_longitude_grids_per_face):
            for j in range(number_of_mid_latitude_grids):
                _r_spherical[:, j, i] = [1.0, lon_centers[i], lat_centers[j]]

                _r_corners_spherical[:, 0, j, i] = [1.0, lon_bounds[i], lat_bounds[j]]
                _r_corners_spherical[:, 1, j, i] = [1.0, lon_bounds[i+1], lat_bounds[j]]
                _r_corners_spherical[:, 2, j, i] = [1.0, lon_bounds[i+1], lat_bounds[j+1]]
                _r_corners_spherical[:, 3, j, i] = [1.0, lon_bounds[i], lat_bounds[j+1]]

        _stack_r_spherical.append(_r_spherical)
        _stack_r_corners_spherical.append(_r_corners_spherical)
 
    def rotate_90deg_along_z_axis(pts):
        return cartesian_to_spherical(rotate_along_z_axis(spherical_to_cartesian(pts), np.pi/2))

    # Construct the other three faces by rotation
    for i in range(3):
        _stack_r_spherical.append(rotate_90deg_along_z_axis(_stack_r_spherical[-1]))
        _stack_r_corners_spherical.append(rotate_90deg_along_z_axis(_stack_r_corners_spherical[-1]))

    # Polar caps
    def _generate_north_cap(lat_critical: float):
    
        # d is the auxilary sqaure that has the side length of 2d
        d = cos(lat_critical) * cos(np.pi/4)
        lon_bounds = - np.pi / 4 + np.linspace(0, np.pi/2, number_of_longitude_grids_per_face+1)
        cap_oneside_bounds = d * np.tan(lon_bounds)
        cap_oneside_centers = (cap_oneside_bounds[1:] + cap_oneside_bounds[:-1]) / 2.0

        def get_r_max(lon):
            return  d / np.cos( (lon + np.pi/4) % (np.pi/2) - np.pi/4 )

        def projected_lon_lat_north_cap(x, y):
            lon = np.atan2(y, x)
            return lon, ( np.pi/2 - (np.pi/2 - lat_critical) * (x**2 + y**2)**0.5 / get_r_max(lon))

        # North cap
        _north_r_spherical = np.zeros((3, number_of_longitude_grids_per_face, number_of_longitude_grids_per_face))
        _north_r_corners_spherical = np.zeros((3, 4, number_of_longitude_grids_per_face, number_of_longitude_grids_per_face))
        for i in range(number_of_longitude_grids_per_face):
            for j in range(number_of_longitude_grids_per_face):
                _north_r_spherical[:, j, i] = [1.0, *projected_lon_lat_north_cap(cap_oneside_centers[i], cap_oneside_centers[j])]
                
                _north_r_corners_spherical[:, 0, j, i] = [1.0, *projected_lon_lat_north_cap(cap_oneside_bounds[i], cap_oneside_bounds[j])]
                _north_r_corners_spherical[:, 1, j, i] = [1.0, *projected_lon_lat_north_cap(cap_oneside_bounds[i+1], cap_oneside_bounds[j])]
                _north_r_corners_spherical[:, 2, j, i] = [1.0, *projected_lon_lat_north_cap(cap_oneside_bounds[i+1], cap_oneside_bounds[j+1])]
                _north_r_corners_spherical[:, 3, j, i] = [1.0, *projected_lon_lat_north_cap(cap_oneside_bounds[i], cap_oneside_bounds[j+1])]

        return _north_r_spherical, _north_r_corners_spherical

    def rotate_to_south(pts):
        return cartesian_to_spherical(rotate_along_x_axis(spherical_to_cartesian(pts), np.pi))
 
    north_r_spherical, north_r_corners_spherical = _generate_north_cap(lat_north_critical)
    south_r_spherical, south_r_corners_spherical = _generate_north_cap(-lat_south_critical)

    _stack_r_spherical.append(north_r_spherical)
    _stack_r_corners_spherical.append(north_r_corners_spherical)

    _stack_r_spherical.append(rotate_to_south(south_r_spherical))
    _stack_r_corners_spherical.append(rotate_to_south(south_r_corners_spherical))

    _stack_grid_numbering = []
    tmp_numbering_count = 0
    # Construct numbering
    for tile, tile_r_spherical in enumerate(_stack_r_spherical):
        nj, ni = tile_r_spherical.shape[1:3]
        _stack_grid_numbering.append(np.arange(nj * ni, dtype=int).reshape((nj, ni)) + tmp_numbering_count)
        tmp_numbering_count += nj * ni

    # Construct land-sea mask
    _stack_binary_mask = []
    for tile, tile_r_spherical in enumerate(_stack_r_spherical):
        lon = tile_r_spherical[1, :] * 180/np.pi
        lat = tile_r_spherical[2, :] * 180/np.pi
        _stack_binary_mask.append(np.ones_like(lon))# globe.is_land( lat,lon ))
        
    # Construct solid angles
    _stack_solid_angle = [
        compute_solid_angle(tile_r_corners_spherical)
        for tile_r_corners_spherical in _stack_r_corners_spherical
    ]

    # Construct neighbor tile relationship
    neighbor_tile_and_rotation = generate_neighbor_tile_and_rotation(number_of_tiles_in_mid_latitude=1)

    return GenericLatLonCap(
        number_of_longitude_grids_per_face = number_of_longitude_grids_per_face,
        number_of_mid_latitude_grids = number_of_mid_latitude_grids,
        r_spherical = (ref := _stack_r_spherical),
        r_corners_spherical = _stack_r_corners_spherical,
        angle_between_grid_north_and_spherical_north = None,
        grid_numbering = _stack_grid_numbering,
        neighbor_grid_numbering = None,
        number_of_tiles = len(ref),
        binary_mask = _stack_binary_mask,
        neighbor_tile_and_rotation = neighbor_tile_and_rotation,
        grid_solid_angles = _stack_solid_angle,
    )

def generate_tiled_latloncap(
    lat_north_critical_degree: float,
    lat_south_critical_degree: float,
    number_of_grids_per_side: int,
    number_of_tiles_in_mid_latitude: int = 3,
):
    
    """
        lat_critical_degree : The latitude (in degree) beyond which polar cap starts.
        number_of_grids_per_side: the size of the tile. Each tile has equal sides. 
    """
  
    generic_llc = generate_generic_latloncap(
        lat_north_critical_degree = lat_north_critical_degree,
        lat_south_critical_degree = lat_south_critical_degree,
        number_of_longitude_grids_per_face = number_of_grids_per_side,
        number_of_mid_latitude_grids = number_of_tiles_in_mid_latitude * number_of_grids_per_side,
    )

    split_ranges = [ slice(i*number_of_grids_per_side, (i+1)*number_of_grids_per_side) for i in range(number_of_tiles_in_mid_latitude) ]
    
    def _split_face_into_tiles(a:np.ndarray, axis: int):
        axis = axis % len(a.shape)
        _padding_axis_front = (slice(None),) * axis
        _padding_axis_back = (slice(None),) * ( len(a.shape) - 1 - axis )
        return [ a[*_padding_axis_front, split_range, *_padding_axis_back ] for split_range in split_ranges ]
   
    def _convert_generic_to_tiles(a: List[np.ndarray], split_axis): 
        if len(a) != 6:
            raise ValueError("Length of input list should be 6 exactly.")
            
        x = _split_face_into_tiles(a[0], axis=split_axis)
        return stack([
            *_split_face_into_tiles(a[0], axis=split_axis),
            *_split_face_into_tiles(a[1], axis=split_axis),
            *_split_face_into_tiles(a[2], axis=split_axis),
            *_split_face_into_tiles(a[3], axis=split_axis),
            a[4],
            a[5],
        ], axis=0)
    
    neighbor_tile_and_rotation = generate_neighbor_tile_and_rotation(number_of_tiles_in_mid_latitude=number_of_tiles_in_mid_latitude)
    return TiledLatLonCap(
        number_of_grids_per_side = number_of_grids_per_side,
        r_spherical = ( ref := _convert_generic_to_tiles(generic_llc.r_spherical, split_axis=-2)),
        r_corners_spherical = _convert_generic_to_tiles(generic_llc.r_corners_spherical, split_axis=-2),
        angle_between_grid_north_and_spherical_north = None,
        grid_numbering = _convert_generic_to_tiles(generic_llc.grid_numbering, split_axis=-2),
        neighbor_grid_numbering = None,
        number_of_tiles = ref.shape[0],
        binary_mask = _convert_generic_to_tiles(generic_llc.binary_mask, split_axis=-2),
        neighbor_tile_and_rotation = neighbor_tile_and_rotation,
        number_of_tiles_in_mid_latitude = number_of_tiles_in_mid_latitude,
        grid_solid_angles = _convert_generic_to_tiles(generic_llc.grid_solid_angles, split_axis=-2),
        shapes = dict(
            T = (ref.shape[0], number_of_grids_per_side, number_of_grids_per_side),
        ),
    )

def construct_halo_pad(a: np.ndarray, tiled_llc: TiledLatLonCap, tile: int):
    
    # determine its neighboring tiles: E, W, S, and N. Including how to rotate them such that
    #                                  the neighboring tiles will be in the aligned direction.                                   
    
    # determine 
    
    pass


def get_padding(a: np.ndarray, tile: int, direction: int, tiled_llc: TiledLatLonCap, number_of_padding:int = 1):
    
    linked_tile_info = tiled_llc.neighbor_tile_and_rotation[tile, direction]
    linked_tile = a[linked_tile_info[0]]

    indexing = None    
    if direction == 0: # EAST
        indexing = (slice(None), slice(None, number_of_padding, None))
    elif direction == 1: # NORTH
        indexing = (slice(None, number_of_padding, None), slice(None))
    elif direction == 2: # WEST
        indexing = (slice(None), slice(- number_of_padding, None, None))
    elif direction == 3: # SOUTH
        indexing = (slice(- number_of_padding, None, None), slice(None))
    
    return np.rot90(a[linked_tile_info[0]], k=linked_tile_info[1], axes=(0, 1))[indexing]
  

def write_to_SCRIP_grid_file(tiled_llc: TiledLatLonCap, output_file: str | Path, flatten: bool = True):
    
    import xarray as xr
   
    grid_size = tiled_llc.binary_mask.size
    grid_corners = 4
    if flatten:
        grid_dims = [ tiled_llc.binary_mask.size ]
    else:
        grid_dims = list(tiled_llc.binary_mask.shape)[::-1]
    
    #grid_dims = list(tiled_llc.binary_mask.shape)[::-1]
    grid_center_lon = tiled_llc.r_spherical[:, 1, :, :].flatten() 
    grid_center_lat = tiled_llc.r_spherical[:, 2, :, :].flatten() 
    
    grid_imask = tiled_llc.binary_mask.flatten()
    
    grid_corner_lat = np.zeros((grid_size, grid_corners))
    grid_corner_lon = np.zeros((grid_size, grid_corners))
    grid_area = tiled_llc.grid_solid_angles.flatten() 
    for i in range(grid_corners): 
        _lat = tiled_llc.r_corners_spherical[:, 2, i, :, :].flatten()
        grid_corner_lon[:, i] = tiled_llc.r_corners_spherical[:, 1, i, :, :].flatten()
        grid_corner_lat[:, i] = _lat
    

    if flatten:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( ["grid_size", ], grid_imask),
                grid_center_lat = ( ["grid_size", ], grid_center_lat, {"units" : "radians"} ),
                grid_center_lon = ( ["grid_size", ], grid_center_lon, {"units" : "radians"} ),
                grid_corner_lat = ( ["grid_size", "grid_corners"], grid_corner_lat, {"units" : "radians"} ),
                grid_corner_lon = ( ["grid_size", "grid_corners"], grid_corner_lon, {"units" : "radians"} ),
                grid_area = ( ["grid_size",], grid_area, {"units" : "radians^2"} ),
            ),
        )
    else:
        # Debug purpose
        tiled_dim_names = ["tile", "j", "i"]
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( [*tiled_dim_names], grid_imask.reshape(grid_dims)),
                grid_center_lat = ( [*tiled_dim_names], grid_center_lat.reshape(grid_dims), {"units" : "radians"} ),
                grid_center_lon = ( [*tiled_dim_names], grid_center_lon.reshape(grid_dims), {"units" : "radians"} ),
                grid_corner_lat = ( [*tiled_dim_names, "grid_corners"], grid_corner_lat.reshape(grid_dims + [grid_corners,]), {"units" : "radians"} ),
                grid_corner_lon = ( [*tiled_dim_names, "grid_corners"], grid_corner_lon.reshape(grid_dims + [grid_corners,]), {"units" : "radians"} ),
                grid_area = ( [*tiled_dim_names], grid_area.reshape(grid_dims), {"units" : "radians^2"} ),
            ),
        )

    ds.to_netcdf(output_file)

def test_check_grid(): 

    generic_llc = generate_generic_latloncap(
        lat_north_critical_degree =  60.0,
        lat_south_critical_degree = -80.0,
        number_of_mid_latitude_grids = 10,
        number_of_longitude_grids_per_face = 5,
    )

    tiled_llc = generate_tiled_latloncap(
        lat_north_critical_degree =  60.0,
        lat_south_critical_degree = -70.0,
        number_of_grids_per_side = 10,
    )

    selected_tile_number = 0
    print("Selected Tile: ", selected_tile_number)
    print(tiled_llc.grid_numbering[selected_tile_number])

    print("East of this tile: ")
    print(tiled_llc.grid_numbering[3])

    print("North of this tile: ")
    print(tiled_llc.grid_numbering[1])

    print("West of this tile: ")
    print(tiled_llc.grid_numbering[9])

    print("South of this tile: ")
    print(tiled_llc.grid_numbering[13])


    for direction in [0, 1, 2, 3]:
        print(f"Direction = {direction}")
        padding = get_padding(
            tiled_llc.grid_numbering,
            tile = selected_tile_number,
            direction = direction,
            number_of_padding = 2,
            tiled_llc = tiled_llc,
        )
        info = tiled_llc.neighbor_tile_and_rotation[selected_tile_number, direction]
        print(f"{selected_tile_number} => {info[0]} (rot90 = {info[1]})")
        print(padding)



    import numpy as np
    import matplotlib.pyplot as plt
    

    print("Generic LLC test: ")

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(azim=-30, elev=45, roll=0) 

    ax.scatter(0, 0, 0, color="red", s=10)
    for tile, (tile_r_spherical_T, tile_r_corners_spherical_T, tile_grid_numbering) in enumerate(zip(generic_llc.r_spherical, generic_llc.r_corners_spherical, generic_llc.grid_numbering)):
        tile_r_cartesian_T = spherical_to_cartesian(tile_r_spherical_T)
        ax.scatter(tile_r_cartesian_T[0, :], tile_r_cartesian_T[1, :], tile_r_cartesian_T[2, :])

        #for x, y, z, numbering in  zip( tile_r_cartesian_T[0].flatten(), tile_r_cartesian_T[1].flatten(), tile_r_cartesian_T[2].flatten(), tile_grid_numbering.flatten()):
        #    ax.text(x, y, z, f"{numbering}")
      
        for i in range(tile_r_corners_spherical_T.shape[2]): 
            for j in range(tile_r_corners_spherical_T.shape[3]):
                if np.random.rand() > 0.85:
                    for k in range(4):
                        p1 = spherical_to_cartesian(tile_r_corners_spherical_T[:, k, i, j])
                        p2 = spherical_to_cartesian(tile_r_corners_spherical_T[:, (k+1)%4, i, j])
                        ax.quiver(*p1, *(p2-p1), color='blue', arrow_length_ratio=0.2, colors=["black", "orange", "blue", "green"][k])

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")
    ax.set_aspect('equal')
    ax.set_title("Generic LLC")
   
    # ============================================
 
    print("Tiled LLC test: ")

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(azim=-30, elev=45, roll=0) 

    ax.scatter(0, 0, 0, color="red", s=10)

    for tile in range(tiled_llc.number_of_tiles):

        tile_r_cartesian_T = spherical_to_cartesian(tiled_llc.r_spherical[tile])
        tile_r_corners_spherical_T = tiled_llc.r_corners_spherical[tile]
        tile_grid_numbering = tiled_llc.grid_numbering[tile]
        tile_binary_mask = tiled_llc.binary_mask[tile]
        
        _tmp = tile_r_cartesian_T[0]
        _tmp[tile_binary_mask == 0] = np.nan
        tile_r_cartesian_T[0, :] = _tmp
        
        ax.scatter(tile_r_cartesian_T[0, :], tile_r_cartesian_T[1, :], tile_r_cartesian_T[2, :])
       
        """ 
        for x, y, z, numbering in  zip( tile_r_cartesian_T[0].flatten(), tile_r_cartesian_T[1].flatten(), tile_r_cartesian_T[2].flatten(), tile_grid_numbering.flatten()):
            ax.text(x, y, z, f"{numbering}")
     
        for i in range(tiled_llc.number_of_grids_per_side): 
            for j in range(tiled_llc.number_of_grids_per_side):
                if np.random.rand() > 0.85:
                    for k in range(4):
                        p1 = spherical_to_cartesian(tile_r_corners_spherical_T[:, k, i, j])
                        p2 = spherical_to_cartesian(tile_r_corners_spherical_T[:, (k+1)%4, i, j])
                        ax.quiver(*p1, *(p2-p1), color='blue', arrow_length_ratio=0.2, colors=["black", "orange", "blue", "green"][k])
        """

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")
    ax.set_aspect('equal')
    ax.set_title("Tiled LLC")

    plt.show()
    

def test_output_SCRIP_file():
    output_file = "tiled_llc.nc"

    print("Generating grid...") 
    tiled_llc = generate_tiled_latloncap(
        lat_north_critical_degree =  60.0,
        lat_south_critical_degree = -70.0,
        number_of_grids_per_side = 20,
    )

    print("Writing to file: ", output_file)
    write_to_SCRIP_grid_file(tiled_llc, output_file, flatten=False)


if __name__ == "__main__":
    test_output_SCRIP_file()
     

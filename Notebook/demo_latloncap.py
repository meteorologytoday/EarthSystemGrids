# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LatLonCap Demo

# %%
import EarthSystemGrids.LatLonCap as LLC
from EarthSystemGrids.LatLonCap import spherical_to_cartesian

# %%
tiled_llc = LLC.generate_tiled_latloncap(
    lat_north_critical_degree =  60.0,
    lat_south_critical_degree = -70.0,
    number_of_grids_per_side = 3,
)

# %%
import numpy as np
import matplotlib
# %matplotlib ipympl
import matplotlib.pyplot as plt

# %%
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
   
    
    for x, y, z, numbering in  zip( tile_r_cartesian_T[0].flatten(), tile_r_cartesian_T[1].flatten(), tile_r_cartesian_T[2].flatten(), tile_grid_numbering.flatten()):
        ax.text(x, y, z, f"{numbering}")
    
    for i in range(tiled_llc.number_of_grids_per_side): 
        for j in range(tiled_llc.number_of_grids_per_side):
            if np.random.rand() > 0.85:
                for k in range(4):
                    p1 = spherical_to_cartesian(tile_r_corners_spherical_T[:, k, i, j])
                    p2 = spherical_to_cartesian(tile_r_corners_spherical_T[:, (k+1)%4, i, j])
                    ax.quiver(*p1, *(p2-p1), color='blue', arrow_length_ratio=0.2, colors=["black", "orange", "blue", "green"][k])
    

ax.set_xlabel("x-direction")
ax.set_ylabel("y-direction")
ax.set_zlabel("z-direction")
ax.set_aspect('equal')
ax.set_title("Tiled LLC")

plt.show()


# %%

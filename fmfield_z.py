import os
import tempfile
import pint

ureg = pint.UnitRegistry()
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 4)

import tdgl
from tdgl.geometry import box, circle
from tdgl.visualization.animate import create_animation
from tdgl.sources import LinearRamp

"""
Vector potential calcuations
"""
from scipy import interpolate
from tdgl import Parameter

# B_Z = np.reshape(B_Z, (np.shape(B_Z)[0] * np.shape(B_Z)[1], 1))

def textured_vector_potential(
    positions,
    Bz,
):
    """
    Calculates the magnetic vector potential [Ax, Ay, Az] at ``positions``
    due uniform magnetic field along the z-axis with strength ``Bz``.

    Args:
    positions: Shape (n, 3) array of (x, y, z) positions in meters at which to
        evaluate the vector potential.
    Bz: The strength of the the field with shape (m, m) with units of Tesla, where
    m is the size of the Mumax simulation

    Returns:
    Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``
    in units of Tesla * meter.

    """
    # assert isinstance(Bz, (float, str, pint.Quantity)), type(Bz)
    # positions = np.atleast_2d(positions)
    # assert positions.shape[1] == 3, positions.shape
    # if not isinstance(positions, pint.Quantity):
    #     positions = positions * ureg("meter")
    # if isinstance(Bz, str):
    #     Bz = ureg(Bz)
    # if isinstance(Bz, float):
    #     Bz = Bz * ureg("tesla")


    # Assuming 'positions' is already defined as in the previous example
    # Extract the x and y values from the positions array

    xy_vals = positions[:, :2]
    
    # Calculate the range (peak-to-peak) of x and y values
    dx = np.ptp(xy_vals[:, 0])
    dy = np.ptp(xy_vals[:, 1])
    # Calculate the center point for x and y
    center_x = np.min(xy_vals[:, 0]) + dx / 2
    center_y = np.min(xy_vals[:, 1]) + dy / 2
    center = np.array([center_x, center_y])
    # Subtract the center point from all xy values to center the data
    xy_vals_centered = xy_vals - center
    centered_xs = xy_vals_centered[:, 0]
    centered_ys = xy_vals_centered[:, 1]
    print("X range:", centered_xs.min(), centered_xs.max())
    print("Y range:", centered_ys.min(), centered_ys.max())
    # make a grid equally sized as the positions but with spacings equivalent to the Mumax mesh
    flattened_Bz_values = np.reshape(Bz, (np.shape(Bz)[0] * np.shape(Bz)[1], 1))
    # Changes
    grid_x = np.linspace(-450*1e-9, 62*1e-9, np.shape(Bz)[0])
    grid_y = np.linspace(-450*1e-9, 62*1e-9, np.shape(Bz)[1])
    X, Y = np.meshgrid(grid_x, grid_y)
    Bz_points = np.vstack([X.ravel(), Y.ravel()]).T
    interpolated_Bz = interpolate.griddata(
        points = Bz_points,
        values = flattened_Bz_values,
        xi = xy_vals_centered,
        method = "linear",
        fill_value = 0.0,  # <- sets Bz=0 outside the grid
    )

    # interpolate to find Bz at positions
    #interpolated_Bz = interpolate.griddata(Bz_points, flattened_Bz_values, xy_vals_centered)
    interpolated_Bz = interpolated_Bz*ureg("tesla")
    centered_ys = centered_ys*ureg("meter")
    centered_xs = centered_xs*ureg("meter")

    # Fix broadcasting by reshaping Bz
    Axy = 1/2*interpolated_Bz * np.stack([-1*centered_ys, centered_xs], axis=1)
    # Add z = 0 component
    A = np.hstack([Axy, np.zeros_like(Axy[:, :1])])
    A = A.to("tesla * meter")
    return A

def FM_field_vector_potential(
    x,
    y,
    z,
    *,
    multiplier,
    field_units: str = "T",
    length_units: str = "nm",
):
    CURRENT_DIRECTORY = os.path.dirname(os.getcwd())
    #################################################
    # CHANGE THIS DEPENDING ON APPLIED FIELD IN RUN #
    #################################################
    #DATA_AND_LAYER_NAME = "0000_full_mag_40mT_layer2.npy"
    #DEMAG_B_Z_FILEPATH = os.path.join("Documents/GitHub/sup-spin/mumax3/kaiyang/Fe3Co7.out/0004_full_mag_90mT_layer2.npy", "%s" % DATA_AND_LAYER_NAME)
    DEMAG_B_Z = np.load("Documents/GitHub/sup-spin/mumax3/kaiyang/Fe3Co7.out/0004_full_mag_90mT_layer2.npy")
    
    #################################################
    # CHANGE THIS DEPENDING ON APPLIED FIELD IN RUN #
    #################################################
    APPLIED_B_Z = 0
    B_Z = (DEMAG_B_Z + APPLIED_B_Z)*multiplier
    if z.ndim == 0:
        z = z * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    positions = (positions * ureg(length_units)).to("m").magnitude
    Bz = B_Z * ureg(field_units)
    A = textured_vector_potential(positions, Bz)
    return A.to(f"{field_units} * {length_units}").magnitude


def FMField(
    multiplier=None,field_units: str = "T", length_units: str = "nm",
) -> Parameter:
    """Returns a Parameter that computes a constant as a function of ``x, y, z``.
    Args:
        value: The constant value of the field.
    Returns:
        A Parameter that returns ``value`` at all ``x, y, z``.
    """
    return Parameter(
        FM_field_vector_potential,
        multiplier=multiplier,
        field_units=field_units,
        length_units=length_units,
    )

    

"""
def FM_field_vector_potential(
    x,
    y,
    z,
    *,
    field_units: str = "T",
    length_units: str = "nm",
):
    if z.ndim == 0:
        z = z * np.ones_like(x)
    
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    positions = (positions * ureg(length_units)).to("m").magnitude
    
    # Set Bz based only on x value: 1 if x < 0, -1 if x > 0
    Bz = np.where(x < 0, 1, -1) * ureg(field_units)

    # Compute the vector potential A using the symmetric gauge
    centered_xs = x * ureg(length_units).to("m")
    centered_ys = y * ureg(length_units).to("m")

    # Compute the vector potential A from Bz (assuming symmetric gauge A = (B/2) * [-y, x])
    Axy = (1/2) * Bz[:, np.newaxis] * np.stack([-1 * centered_ys, centered_xs], axis=1)

    # Append zero for Az component
    A = np.hstack([Axy, np.zeros_like(Axy[:, :1])])

    return A.to(f"{field_units} * {length_units}").magnitude
"""
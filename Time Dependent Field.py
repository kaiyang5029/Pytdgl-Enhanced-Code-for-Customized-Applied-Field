"""
@author: Kaiyang
This code gives an example to implement time-dependent field.
The time-dependent function should be defined in field_vector_potential
In pytdgl, just call "applied_vector_potential = Field" will do
"""

import numpy as np

from tdgl.em import uniform_Bz_vector_potential, ureg
from tdgl import Parameter


def field_vector_potential(
    x,
    y,
    z,
    *,
    t,
    field_units: str = "mT",
    length_units: str = "nm",
):
    if z.ndim == 0:
        z = z * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    positions = (positions * ureg(length_units)).to("m").magnitude
    Bz = (int(t/100)%2)*50
    Bz = Bz * ureg(field_units)
    A = uniform_Bz_vector_potential(positions, Bz)
    return A.to(f"{field_units} * {length_units}").magnitude


def Field(
    value: float = 0, field_units: str = "mT", length_units: str = "nm"
) -> Parameter:
    """Returns a Parameter that computes a constant as a function of ``x, y, z``.
    Args:
        value: The constant value of the field.
    Returns:
        A Parameter that returns ``value`` at all ``x, y, z``.
    """
    return Parameter(
        field_vector_potential,
        field_units=field_units,
        length_units=length_units,
        time_dependent=True,
    )
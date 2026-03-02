"""
@author: Justin Chen (m_outline), Kai Yang TAN (vorticity)
This code tries to incorporate magnetic field outline (m_outline) into the visualization.

Follow the examples given in the "Ramping Field.ipynb"
First import all functions "make_video_from_solution","plot_order_parameter","plot_vorticity"
Define m_outline
Call the function with both solutions and m_outline.
"""
import logging
import os
from contextlib import nullcontext
from logging import Logger
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import h5py
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm

from tdgl.device.device import Device
from tdgl.solution.data import get_data_range
from tdgl.visualization.common import DEFAULT_QUANTITIES, Quantity, auto_grid
from tdgl.visualization.io import get_plot_data, get_state_string
import tdgl

from IPython.display import HTML, display
from tdgl import Solution
import matplotlib as mpl

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

x_location = [-450,62]
y_location = [-450,62]

@dataclass
class PlotDefault:
    cmap: str
    clabel: str
    xlabel: str = "$x/\\xi$"
    ylabel: str = "$y/\\xi$"
    vmin: Union[float, None] = None
    vmax: Union[float, None] = None
    symmetric: bool = False
    
PLOT_DEFAULTS = {
    Quantity.ORDER_PARAMETER: PlotDefault(
        cmap="viridis", clabel="$|\\psi|$", vmin=0, vmax=1
    ),
    Quantity.PHASE: PlotDefault(
        cmap="twilight_shifted", clabel="$\\arg(\\psi)/\\pi$", vmin=-1, vmax=1
    ),
    Quantity.SUPERCURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_s|/J_0$"),
    Quantity.NORMAL_CURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_n|/J_0$"),
    Quantity.SCALAR_POTENTIAL: PlotDefault(cmap="magma", clabel="$\\mu/v_0$"),
    Quantity.APPLIED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{applied}}/(\\xi B_{{c2}})$"
    ),
    Quantity.INDUCED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{induced}}/(\\xi B_{{c2}})$"
    ),
    Quantity.EPSILON: PlotDefault(
        cmap="viridis", clabel="$\\epsilon$", vmin=-1, vmax=1
    ),
    Quantity.VORTICITY: PlotDefault(
        cmap="coolwarm",
        clabel="$(\\vec{{\\nabla}}\\times\\vec{{J}})\\cdot\\hat{{z}}$",
        symmetric=True,vmin=-0.01, vmax=0.01
    ),
}

def create_animation(
    input_file: Union[str, h5py.File],
    m_outline_filename,
    *,
    output_file: Optional[str] = None,
    quantities: Union[str, Sequence[str]] = DEFAULT_QUANTITIES,
    shading: Literal["flat", "gouraud"] = "gouraud",
    fps: int = 30,
    dpi: float = 100,
    max_cols: int = 4,
    min_frame: int = 0,
    max_frame: int = -1,
    autoscale: bool = False,
    dimensionless: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    axis_labels: bool = False,
    axes_off: bool = False,
    title_off: bool = False,
    full_title: bool = True,
    logger: Optional[Logger] = None,
    figure_kwargs: Optional[Dict[str, Any]] = None,
    writer: Union[str, animation.MovieWriter, None] = None,
) -> animation.FuncAnimation:
    """Generates, and optionally saves, and animation of a TDGL simulation.

    Args:
        input_file: An open h5py file or a path to an H5 file containing
            the :class:`tdgl.Solution` you would like to animate.
        output_file: A path to which to save the animation,
            e.g., as a gif or mp4 video.
        quantities: The names of the quantities to animate.
        shading: Shading method, "flat" or "gouraud". See matplotlib.pyplot.tripcolor.
        fps: Frame rate in frames per second.
        dpi: Resolution in dots per inch.
        max_cols: The maxiumum number of columns in the subplot grid.
        min_frame: The first frame of the animation.
        max_frame: The last frame of the animation.
        autoscale: Autoscale colorbar limits at each frame.
        dimensionless: Use dimensionless units for axes
        xlim: x-axis limits
        ylim: y-axis limits
        axes_off: Turn off the axes for each subplot.
        title_off: Turn off the figure suptitle.
        full_title: Include the full "state" for each frame in the figure suptitle.
        figure_kwargs: Keyword arguments passed to ``plt.subplots()`` when creating
            the figure.
        writer: A :class:`matplotlib.animation.MovieWriter` instance to use when
            saving the animation.
        logger: A logger instance to use.

    Returns:
        The animation as a :class:`matplotlib.animation.FuncAnimation`.
    """
    if isinstance(input_file, str):
        input_file = input_file
    if quantities is None:
        quantities = Quantity.get_keys()
    if isinstance(quantities, str):
        quantities = [quantities]
    quantities = [Quantity.from_key(name.upper()) for name in quantities]
    num_plots = len(quantities)
    logger = logger or logging.getLogger()
    figure_kwargs = figure_kwargs or dict()
    figure_kwargs.setdefault("constrained_layout", True)
    default_figsize = (
        3.25 * min(max_cols, num_plots),
        2.5 * max(1, num_plots // max_cols),
    )
    figure_kwargs.setdefault("figsize", default_figsize)
    figure_kwargs.setdefault("sharex", True)
    figure_kwargs.setdefault("sharey", True)

    logger.info(f"Creating animation for {[obs.name for obs in quantities]!r}.")

    mpl_context = nullcontext() if output_file is None else plt.ioff()
    if isinstance(input_file, str):
        h5_context = h5py.File(input_file, "r")
    else:
        h5_context = nullcontext(input_file)

    with h5_context as h5file:
        with mpl_context:
            device = Device.from_hdf5(h5file["solution/device"])
            mesh = device.mesh
            if dimensionless:
                scale = 1
                units_str = "\\xi"
            else:
                scale = device.layer.coherence_length
                units_str = f"{device.ureg(device.length_units).units:~L}"
            x, y = scale * mesh.sites.T

            # Get the ranges for the frame
            _min_frame, _max_frame = get_data_range(h5file)
            min_frame = max(min_frame, _min_frame)
            if max_frame == -1:
                max_frame = _max_frame
            else:
                max_frame = min(max_frame, _max_frame)

            # Temp data to use in plots
            temp_value = np.ones(len(mesh.sites), dtype=float)
            temp_value[0] = 0
            temp_value[1] = 0.5

            fig, axes = auto_grid(num_plots, max_cols=max_cols, **figure_kwargs)
            collections = []
            for quantity, ax in zip(quantities, axes.flat):
                ax: plt.Axes
                opts = PLOT_DEFAULTS[quantity]
                collection = ax.tripcolor(
                    x,
                    y,
                    temp_value,
                    triangles=mesh.elements,
                    shading=shading,
                    cmap=opts.cmap,
                    vmin=opts.vmin,
                    vmax=opts.vmax,
                )
                cbar = fig.colorbar(collection, ax=ax)
                cbar.set_label(opts.clabel)
                ax.set_aspect("equal")
                ax.set_title(quantity.value)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if axes_off:
                    ax.axis("off")
                if axis_labels:
                    ax.set_xlabel(f"$x$ [${units_str}$]")
                    ax.set_ylabel(f"$y$ [${units_str}$]")
                collections.append(collection)

            vmins = [+np.inf for _ in quantities]
            vmaxs = [-np.inf for _ in quantities]

            def update(frame):
                if not h5file:
                    return
                frame += min_frame
                state = get_state_string(h5file, frame, max_frame)
                if not full_title:
                    state = state.split(",")[0]
                if not title_off:
                    fig.suptitle(state)
                for i, (quantity, collection) in enumerate(
                    zip(quantities, collections)
                ):
                    opts = PLOT_DEFAULTS[quantity]
                    values, direction, _ = get_plot_data(h5file, mesh, quantity, frame)
                    mask = np.abs(values - np.mean(values)) <= 6 * np.std(values)
                    if opts.vmin is None:
                        if autoscale:
                            vmins[i] = np.min(values[mask])
                        else:
                            vmins[i] = min(vmins[i], np.min(values[mask]))
                    else:
                        vmins[i] = opts.vmin
                    if opts.vmax is None:
                        if autoscale:
                            vmaxs[i] = np.max(values[mask])
                        else:
                            vmaxs[i] = max(vmaxs[i], np.max(values[mask]))
                    else:
                        vmaxs[i] = opts.vmax
                    if opts.symmetric:
                        vmax = max(abs(vmins[i]), abs(vmaxs[i]))
                        vmaxs[i] = vmax
                        vmins[i] = -vmax
                    if shading == "flat":
                        # https://stackoverflow.com/questions/40492511/set-array-in-tripcolor-bug
                        values = values[mesh.elements].mean(axis=1)
                    collection.set_array(values)
                    collection.set_clim(vmins[i], vmaxs[i])
                
                m_outline = np.load(m_outline_filename)
                X, Y = np.meshgrid(np.linspace(x_location[0], x_location[1], m_outline.shape[1]), np.linspace(y_location[1], y_location[0], m_outline.shape[0]))
                if frame == min_frame:
                    for i, ax in enumerate(axes):
                        if i==0:
                            ax.contour(X, Y, m_outline, colors='red', alpha=0.05)
                        else:
                            ax.contour(X, Y, m_outline, colors='black', alpha=0.05)
                fig.canvas.draw()

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=max_frame - min_frame,
                interval=1e3 / fps,
                blit=False,
            )

        if output_file is not None:
            output_file = os.path.join(os.getcwd(), output_file)
            if writer is None:
                kwargs = dict(fps=fps)
            else:
                kwargs = dict(writer=writer)
            fname = os.path.basename(output_file)
            with tqdm(
                total=len(range(min_frame, max_frame)),
                unit="frames",
                desc=f"Saving to {fname}",
            ) as pbar:
                anim.save(
                    output_file,
                    dpi=dpi,
                    progress_callback=lambda frame, total: pbar.update(1),
                    **kwargs,
                )

        return anim

    
def make_video_from_solution(
    solution,
    m_outline_filename,
    quantities=("order_parameter", "phase"),
    fps=20,
    figsize=(5, 4),
):
    """Generates an HTML5 video from a tdgl.Solution."""
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                m_outline_filename=m_outline_filename,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            video = anim.to_html5_video()
        return HTML(video)
    
def setup_color_limits(
    dict_of_arrays: Dict[str, np.ndarray],
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Set up color limits (vmin, vmax) for a dictionary of numpy arrays.

    Args:
        dict_of_arrays: Dict of ``{name: array}`` for which to compute color limits.
        vmin: If provided, this vmin will be used for all arrays. If vmin is not None,
            then vmax must also not be None.
        vmax: If provided, this vmax will be used for all arrays. If vmax is not None,
            then vmin must also not be None.
        share_color_scale: Whether to force all arrays to share the same color scale.
            This option is ignored if vmin and vmax are provided.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
            This option is ignored if vmin and vmax are provided.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.

    Returns:
        A dict of ``{name: (vmin, vmax)}``
    """
    if (vmin is not None and vmax is None) or (vmax is not None and vmin is None):
        raise ValueError("If either vmin or max is provided, both must be provided.")
    if vmin is not None:
        return {name: (vmin, vmax) for name in dict_of_arrays}

    if auto_range_cutoff is None:
        clims = {
            name: (np.nanmin(array), np.nanmax(array))
            for name, array in dict_of_arrays.items()
        }
    else:
        clims = {
            name: auto_range_iqr(array, cutoff_percentile=auto_range_cutoff)
            for name, array in dict_of_arrays.items()
        }

    if share_color_scale:
        # All subplots share the same color scale
        global_vmin = np.inf
        global_vmax = -np.inf
        for vmin, vmax in clims.values():
            global_vmin = min(vmin, global_vmin)
            global_vmax = max(vmax, global_vmax)
        clims = {name: (global_vmin, global_vmax) for name in dict_of_arrays}

    if symmetric_color_scale:
        # Set vmin = -vmax
        new_clims = {}
        for name, (vmin, vmax) in clims.items():
            new_vmax = max(vmax, -vmin)
            new_clims[name] = (-new_vmax, new_vmax)
        clims = new_clims

    return clims

    
def plot_vorticity(
    solution: Solution,
    m_outline_filename,
    ax: Union[plt.Axes, None] = None,
    cmap: str = "coolwarm",
    units: Union[str, None] = None,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    symmetric_color_scale: bool = True,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    shading: str = "gouraud",
    **kwargs,
):
    """Plots the vorticity in the film:
    :math:`\\mathbf{\\omega}=\\mathbf{\\nabla}\\times\\mathbf{K}`.

    .. seealso:

        :meth:`tdgl.Solution.plot_vorticity`

    Args:
        solution: The solution for which to plot the vorticity.
        ax: Matplotlib axes on which to plot.
        cmap: Name of the matplotlib colormap to use.
        units: The units in which to plot the vorticity. Must have dimensions of
            [current] / [length]^2.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

    Returns:
        matplotlib Figure and and Axes.
    """
    if ax is None:
        kwargs.setdefault("constrained_layout", True)
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax.set_aspect("equal")
    device = solution.device
    points = device.points
    triangles = device.triangles
    length_units = device.ureg(device.length_units).units
    if units is None:
        units = solution.vorticity.units
    else:
        units = device.ureg(units)
    v = solution.vorticity.to(units).m
    clim = setup_color_limits(
        {"v": v},
        vmin=vmin,
        vmax=vmax,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )["v"]
    vmin, vmax = clim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    x, y = points[:, 0], points[:, 1]
    im = ax.tripcolor(
        x,
        y,
        v,
        triangles=triangles,
        cmap=cmap,
        norm=norm,
        shading=shading,
    )
    m_outline = np.load(m_outline_filename)
    X, Y = np.meshgrid(np.linspace(x_location[0], x_location[1], m_outline.shape[1]), np.linspace(y_location[1], y_location[0], m_outline.shape[0]))
    ax.contour(X, Y, m_outline, colors='black', alpha=0.1)
    fig.canvas.draw()
    cbar = ax.figure.colorbar(im, cax=cax)
    #cbar = fig.colorbar(im, ax=ax)
    ax.set_title("$\\vec{\\omega}=\\vec{\\nabla}\\times\\vec{K}$")
    ax.set_aspect("equal")
    ax.set_xlabel(f"$x$ [${length_units:~L}$]")
    ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    cbar.set_label(f"$\\vec{{\\omega}}\\cdot\\hat{{z}}$ [${units:~L}$]")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    fig.colorbar(im,cax=cax,fraction=0.046, pad=0.04)
    return fig, ax

def plot_order_parameter(
    solution: Solution,
    m_outline_filename,
    squared: bool = False,
    mag_cmap: str = "viridis",
    phase_cmap: str = "twilight_shifted",
    shading: str = "gouraud",
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots the magnitude (or the magnitude squared) and
    phase of the complex order parameter, :math:`\\psi=|\\psi|e^{i\\theta}`.

    .. seealso:

        :meth:`tdgl.Solution.plot_order_parameter`

    Args:
        solution: The solution for which to plot the order parameter.
        squared: Whether to plot the magnitude squared, :math:`|\\psi|^2`.
        mag_cmap: Name of the colormap to use for the magnitude.
        phase_cmap: Name of the colormap to use for the phase.
        shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

    Returns:
        matplotlib Figure and an array of two Axes objects.
    """
    kwargs.setdefault("figsize", (8, 3))
    kwargs.setdefault("constrained_layout", True)
    device = solution.device
    psi = solution.tdgl_data.psi
    mag = np.abs(psi)
    psi_label = "$|\\psi|$"
    if squared:
        mag = mag**2
        psi_label = "$|\\psi|^2$"
    phase = np.angle(psi) / np.pi
    points = device.points
    triangles = device.triangles
    fig, axes = plt.subplots(1, 2, **kwargs)
    x, y = points[:, 0], points[:, 1]
    m_outline = np.load(m_outline_filename)
    X, Y = np.meshgrid(np.linspace(x_location[0], x_location[1], m_outline.shape[1]), np.linspace(y_location[1], y_location[0], m_outline.shape[0]))
    axes[0].contour(X, Y, m_outline, colors='red', alpha=0.1)
    fig.canvas.draw()
    im = axes[0].tripcolor(
        points[:, 0],
        points[:, 1],
        mag,
        triangles=triangles,
        vmin=0,
        vmax=1,
        cmap=mag_cmap,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label(psi_label)
    im = axes[1].tripcolor(
        points[:, 0],
        points[:, 1],
        phase,
        triangles=triangles,
        vmin=-1,
        vmax=1,
        cmap=phase_cmap,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label("$\\theta / \\pi$")
    length_units = device.ureg(device.length_units).units
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    return fig, axes
#doing relative imports
import sys # for relatove imports of sigma
sys.path.insert(0,"../..")



from sigma.utils import visualisation as visual
from sigma.src.segmentation import PixelSegmenter
from sigma.utils.load import SEMDataset, IMAGEDataset, PIXLDataset
from sigma.utils.loadtem import TEMDataset
from sigma.src.utils import k_factors_120kV
from sigma.utils import normalisation as norm


import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import hyperspy.api as hs
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
from skimage.transform import resize
import seaborn as sns
import altair as alt
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display

# to make sure the plot function works
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)


def search_energy_peak():
    text = widgets.BoundedFloatText(
        value=1.4898, step=0.1, description="Energy (keV):", continuous_update=True
    )
    button = widgets.Button(description="Search")
    out = widgets.Output()

    def button_evenhandler(_):
        out.clear_output()
        with out:
            print("Candidates:")
            print(
                hs.eds.get_xray_lines_near_energy(
                    energy=text.value, only_lines=["a", "b"]
                )
            )

    button.on_click(button_evenhandler)
    widget_set = widgets.HBox([text, button])
    display(widget_set)
    display(out)


def save_fig(fig):
    file_name = widgets.Text(
        value="figure_name.tif",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    folder_name = widgets.Text(
        value="results",
        placeholder="Type something",
        description="Folder name:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    dpi = widgets.BoundedIntText(
        value="96",
        min=0,
        max=300,
        step=1,
        description="Set dpi:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    pad = widgets.BoundedFloatText(
        value="0.01",
        min=0.0,
        description="Set pad:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )
    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            if not os.path.isdir(folder_name.value):
                os.mkdir(folder_name.value)
            if isinstance(fig, mpl.figure.Figure):
                save_path = os.path.join(folder_name.value, file_name.value)
                fig.savefig(
                    save_path, dpi=dpi.value, bbox_inches="tight", pad_inches=pad.value
                )
                print("save figure to", file_name.value)
            else:
                initial_file_name = file_name.value.split(".")
                folder_for_fig = os.path.join(folder_name.value, initial_file_name[0])
                if not os.path.isdir(folder_for_fig):
                    os.mkdir(folder_for_fig)
                for i, single_fig in enumerate(fig):
                    save_path = os.path.join(
                        folder_for_fig,
                        f"{initial_file_name[0]}_{i:02}.{initial_file_name[1]}",
                    )
                    single_fig.savefig(
                        save_path,
                        dpi=dpi.value,
                        bbox_inches="tight",
                        pad_inches=pad.value,
                    )
                print("save all figure to folder:", folder_for_fig)

    button.on_click(save_to)
    all_widgets = widgets.HBox(
        [folder_name, file_name, dpi, pad, button], layout=Layout(width="auto")
    )
    display(all_widgets)
    display(out)


def pick_color(plot_func, *args, **kwargs):
    # Create initial color codes
    hsv = plt.get_cmap("hsv")
    colors = []
    for i in range(len(kwargs["element_list"])):
        colors.append(mpl.colors.to_hex(hsv(i / len(kwargs["element_list"]))[:3]))
    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for element, color in zip(kwargs["element_list"], colors):
        color_pickers.append(
        widgets.ColorPicker(value=color, description=element, layout=layout_format)        )
    # Create an ouput object
    out = widgets.Output()
    with out:
        fig = visual.plot_intensity_maps(**kwargs)
        save_fig(fig)
        
    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                if not isinstance(color_picker, widgets.Button):
                    color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            fig = visual.plot_intensity_maps(**kwargs, colors=color_for_map)
            save_fig(fig)

    button = widgets.Button(description="Set", layout=Layout(width="auto"))
    button.on_click(change_color)

    # Set cmap = viridis for all maps
    text_color = widgets.Text(
        value="viridis",
        placeholder="Type something",
        description="Color map:",
        disabled=False,
        continuous_update=True,
        layout=Layout(width="auto"),
    )

    def set_single_cmap(_):
        out.clear_output()
        with out:
            fig = visual.plot_intensity_maps(**kwargs, colors=text_color.value)
            save_fig(fig)

    button2 = widgets.Button(description="Set color map", layout=Layout(width="auto"))
    button2.on_click(set_single_cmap)

    # Reset button
    def reset(_):
        out.clear_output()
        with out:
            fig = visual.plot_intensity_maps(**kwargs, colors=[])
            save_fig(fig)

    button3 = widgets.Button(description="Reset", layout=Layout(width="auto"))
    button3.on_click(reset)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    color_list.append(button)
    colorpicker_col = widgets.VBox(
        color_list, layout=Layout(flex="8 1 0%", width="80%")
    )
    button_col = widgets.VBox(
        [text_color, button2, button3], layout=Layout(flex="2 1 0%", width="20%")
    )
    color_box = widgets.HBox([colorpicker_col, button_col])

    out_box = widgets.Box([out])

    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def view_dataset(dataset:Union[SEMDataset, TEMDataset, IMAGEDataset], search_energy=True):
    """
    GUI for visualisation of the dataset.
    Shows the navigation image, the summed EDX spectum from all pixels in the dataset, and elemental maps for all features in feature list.

    Includes the ability to search for X-Ray peaks by energy, and add X-Ray lines to the feature list, all within the GUI.

    Tabs for visualisation include:
    Navigation Image
    Summed spectra
    Raw feature maps
    Binned feature maps (if the raw data has been binned and/or normalised)

    Parameters
    ----------
    dataset : SEMDataset, TEMDataset, or IMAGEDataset
              A SEM/STEM EDX dataset
    search_energy : bool
                    Adds the ability to search for X-Ray peaks by energy within the GUI


    """
    if search_energy == True:
        search_energy_peak()

    nav_img_out = widgets.Output()
    with nav_img_out:
        dataset.nav_img.plot(colorbar=False)
        plt.show()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(dataset.nav_img.data, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.close()

    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(dataset.spectra)

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        pick_color(
            visual.plot_intensity_maps, spectra=dataset.spectra, element_list=dataset.feature_list
        )
        # fig = visual.plot_intensity_maps(sem.spectra, sem.feature_list)
        # save_fig(fig)

    if dataset.spectra_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            pick_color(
                visual.plot_intensity_maps,
                spectra=dataset.spectra_bin,
                element_list=dataset.feature_list,
            )
            # fig = visual.plot_intensity_maps(sem.spectra_bin, sem.feature_list)
            # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(dataset.feature_list):
        if i == len(dataset.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="400px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            dataset.set_feature_list(feature_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(dataset.spectra)

        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(dataset.spectra, dataset.feature_list)

        if dataset.spectra_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(dataset.spectra_bin, dataset.feature_list)

    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [nav_img_out, sum_spec_out, elemental_map_out]
    if dataset.spectra_bin is not None:
        tab_list += [elemental_map_out_bin]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "Navigation Signal")
    tab.set_title(1, "Sum spectrum")
    tab.set_title(2, "Elemental maps (raw)")
    i = 2
    if dataset.spectra_bin is not None:
        tab.set_title(i + 1, "Elemental maps (binned)")
    display(tab)

def view_im_dataset(im):
    intensity_out = widgets.Output()
    with intensity_out:
        fig, axs = plt.subplots(1, 1)
        axs.imshow(im.intensity_map, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.show()

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        pick_color(
            visual.plot_intensity_maps, 
            spectra=im.chemical_maps, 
            element_list=im.feature_list
        )
        # fig = visual.plot_intensity_maps(sem.spectra, sem.feature_list)
        # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(im.feature_list):
        if i == len(im.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="400px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            im.set_feature_list(feature_list)

        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(im.chemical_maps, im.feature_list)


    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [intensity_out, elemental_map_out]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "Intensity image")
    tab.set_title(1, "Elemental maps (raw)")
    display(tab)


def view_rgb(dataset:Union[SEMDataset,TEMDataset,IMAGEDataset]):
    """
    Function to plot the intensities of up to three X-Ray features as an RGB map.
    The feature for each colour channel is chosen from a drop down menu within the GUI.

    Parameters
    ----------
    dataset : SEM/TEM/IMAGE dataset containing X_ray features and (normalised) intensities to be plotted in the RGB image.


    """
    
    option_dict = {}
    if isinstance(dataset.normalised_elemental_data, np.ndarray):
        option_dict["normalised"] = dataset.normalised_elemental_data
        option_dict["normalised"]/= option_dict["normalised"].max(keepdims=True, axis=(0,1))

    if type(dataset) != IMAGEDataset:
        option_dict["binned"] = dataset.get_feature_maps()
        option_dict["binned"] = option_dict["binned"].clip(None, np.percentile(option_dict["binned"], 99, axis=(0,1), keepdims=True))
        option_dict["binned"] /= option_dict["binned"].max(keepdims=True, axis=(0,1))
        
        option_dict["raw"] = dataset.get_feature_maps(raw_data=True)
        option_dict["raw"] = option_dict["raw"].clip(None, np.percentile(option_dict["raw"], 99, axis=(0,1), keepdims=True))
        option_dict["raw"] /= option_dict["raw"].max(keepdims=True, axis=(0,1))
    else:
        option_dict["raw"] = dataset.chemical_maps
        option_dict["raw"] = option_dict["raw"].clip(None, np.percentile(option_dict["raw"], 99, axis=(0,1), keepdims=True))
        option_dict["raw"] /= option_dict["raw"].max(keepdims=True, axis=(0,1))
        if dataset.chemical_maps_bin is not None:
            option_dict["binned"] = dataset.chemical_maps_bin
            option_dict["binned"] = option_dict["binned"].clip(None, np.percentile(option_dict["binned"], 99, axis=(0,1), keepdims=True))
            option_dict["binned"] /= option_dict["binned"].max(keepdims=True, axis=(0,1))

    dropdown_option = widgets.Dropdown(
        options=list(option_dict.keys()), description="Data:"
    )
    dropdown_r = widgets.Dropdown(options=dataset.feature_list, description="Red:")
    dropdown_g = widgets.Dropdown(options=dataset.feature_list, description="Green:")
    dropdown_b = widgets.Dropdown(options=dataset.feature_list, description="Blue:")

    plots_output = widgets.Output()

    def dropdown_option_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[change.new],
                elements=[dropdown_r.value, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_r_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[change.new, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_g_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[dropdown_r.value, change.new, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_b_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_rgb(
                dataset,
                elemental_maps=option_dict[dropdown_option.value],
                elements=[dropdown_r.value, dropdown_g.value, change.new],
            )
            save_fig(fig)

    dropdown_option.observe(dropdown_option_eventhandler, names="value")
    dropdown_r.observe(dropdown_r_eventhandler, names="value")
    dropdown_g.observe(dropdown_g_eventhandler, names="value")
    dropdown_b.observe(dropdown_b_eventhandler, names="value")
    color_box = widgets.VBox([dropdown_r, dropdown_g, dropdown_b])
    box = widgets.HBox([dropdown_option, color_box])

    display(box)
    display(plots_output)


def view_pixel_distributions(dataset:Union[SEMDataset, TEMDataset, IMAGEDataset], norm_list:List=[norm.neighbour_averaging,norm.zscore,norm.softmax], cmap:str="viridis"):
    """
    GUI for visualisation of pixel distributions as a result of normalisation processes.
    Shows the intensity distribututions for the chosen feauture in the feature list after each normalisation step.
    The X-Ray line to be imaged is chosen from a drop down menu within the GUI.

    Tabs for visualisation include:
    Navigation Image
    Summed spectra
    Raw feature maps
    Binned feature maps (if the raw data has been binned and/or normalised)

    Parameters
    ----------
    dataset   : SEMDataset, TEMDataset, or IMAGEDataset
                Dataset containing the data cube that has been normalised
       
    norm_list : list
                List of normalisation functions from the `normalisation.py` script. These should be in the order the data was normalised in. 
                Default, [norm.neighbour_averaging,norm.zscore,norm.softmax] as this is the standard normalisation procedure
                The data after each of these steps will be plotted in the gui
                
             


    """
    peak_options = dataset.feature_list
    dropdown_peaks = widgets.Dropdown(options=peak_options, description="Element:")
    
    plots_output = widgets.Output()
    
    with plots_output:
        fig = visual.plot_pixel_distributions(
            dataset=dataset, norm_list=norm_list, peak=dropdown_peaks.value, cmap=cmap
            )
        plt.show()
        save_fig(fig)
            
    def dropdown_option_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = visual.plot_pixel_distributions(
            dataset=dataset, norm_list=norm_list, peak=dropdown_peaks.value, cmap=cmap
            )
            plt.show()
            save_fig(fig)
    
    dropdown_peaks.observe(dropdown_option_eventhandler, names="value")
    out_box = widgets.VBox([dropdown_peaks, plots_output])    
    display(out_box)


def view_intensity_maps(spectra, element_list):
    pick_color(visual.plot_intensity_maps, spectra=spectra, element_list=element_list)


def view_bic(
    latent: np.ndarray,
    n_components: int = 20,
    model: str = "BayesianGaussianMixture",
    model_args: Dict = {"random_state": 6},
):
    bic_list = PixelSegmenter.bic(latent, n_components, model, model_args)
    fig = go.Figure(
        data=go.Scatter(
            x=np.arange(1, n_components + 1, dtype=int),
            y=bic_list,
            mode="lines+markers",
        ),
        layout=go.Layout(
            title="",
            title_x=0.5,
            xaxis_title="Number of component",
            yaxis_title="BIC",
            width=800,
            height=600,
        ),
    )

    fig.update_layout(showlegend=False)
    fig.update_layout(template="simple_white")
    fig.update_traces(marker_size=12)
    fig.show()
    save_csv(pd.DataFrame(data={"bic": bic_list}))


def view_latent_space(ps: PixelSegmenter, color=True):
    colors = []
    cmap = plt.get_cmap(ps.color_palette)
    for i in range(ps.n_components):
        colors.append(mpl.colors.to_hex(cmap(i * (ps.n_components - 1) ** -1)[:3]))

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = []
    for i, c in enumerate(colors):
        color_pickers.append(
            widgets.ColorPicker(
                value=c, description=f"cluster_{i}", layout=layout_format
            )
        )

    newcmp = mpl.colors.ListedColormap(colors, name="new_cmap")
    out = widgets.Output()
    with out:
        fig = ps.plot_latent_space(color=color, cmap=None)
        plt.show()
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        with out:
            color_for_map = []
            for color_picker in color_pickers:
                color_for_map.append(mpl.colors.to_rgb(color_picker.value)[:3])
            newcmp = mpl.colors.ListedColormap(color_for_map, name="new_cmap")
            ps.set_color_palette(newcmp)
            fig = ps.plot_latent_space(color=color, cmap=newcmp)
            save_fig(fig)

    button = widgets.Button(
        description="Set", layout=Layout(flex="8 1 0%", width="auto")
    )
    button.on_click(change_color)

    # Reset button
    def reset(_):
        out.clear_output()
        with out:
            fig = ps.plot_latent_space(color=color, cmap=ps._color_palette)
            save_fig(fig)

    button2 = widgets.Button(
        description="Reset", layout=Layout(flex="2 1 0%", width="auto")
    )
    button2.on_click(reset)

    color_list = []
    for row in range((len(color_pickers) // 5) + 1):
        color_list.append(widgets.HBox(color_pickers[5 * row : (5 * row + 5)]))

    button_box = widgets.HBox([button, button2])
    color_box = widgets.VBox(
        [widgets.VBox(color_list), button_box],
        layout=Layout(flex="2 1 0%", width="auto"),
    )
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))
    final_box = widgets.VBox([color_box, out_box])
    display(final_box)


def plot_latent_density(ps: PixelSegmenter, bins=50):
    z = np.histogram2d(x=ps.latent[:, 0], y=ps.latent[:, 1], bins=bins)[0]
    sh_0, sh_1 = z.shape
    x, y = (
        np.linspace(ps.latent[:, 0].min(), ps.latent[:, 0].max(), sh_0),
        np.linspace(ps.latent[:, 1].min(), ps.latent[:, 1].max(), sh_1),
    )
    fig = go.Figure(data=[go.Surface(z=z.T, x=x, y=y, colorscale="RdBu_r")])
    fig.update_layout(
        title="Density of pixels in latent space",
        autosize=True,
        width=500,
        height=500,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig.show()


def check_latent_space(ps: PixelSegmenter, ratio_to_be_shown=0.25, show_map=False):
    #creating colours
    # Use recoloured cluster colors from interactive_latent_plot if available, otherwise fallback to default palette
    if hasattr(ps, 'cluster_colors') and ps.cluster_colors:
        # Custom user-assigned colors
        domain = sorted(set(ps.labels.flatten()))
        range_ = []
        
        # Assign custom colors where available, fallback to colormap if not
        for cid in domain:
            if cid in ps.cluster_colors:
                range_.append(ps.cluster_colors[cid])  # Use manual color if available
            else:
                # Generate a color from the colormap for clusters that have not been recolored
                cmap = cm.get_cmap(ps.color_palette)
                r, g, b = cmap(cid * (ps.n_components - 1) ** -1)[:3]
                hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
                range_.append(hex_color)

    else:
        # Default colormap-based colors
        cmap = cm.get_cmap(ps.color_palette)
        domain = [i for i in range(ps.n_components)]
        range_ = []
        
        # Generate colors from the colormap for all clusters
        for i in range(ps.n_components):
            r, g, b = cmap(i * (ps.n_components - 1) ** -1)[:3]
            hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            range_.append(hex_color)


    latent, dataset, feature_list, labels = (
        ps.latent,
        ps.dataset.normalised_elemental_data,
        ps.dataset.feature_list,
        ps.labels,
    )
    x_id, y_id = np.meshgrid(range(ps.width), range(ps.height))
    x_id = x_id.ravel().reshape(-1, 1)
    y_id = y_id.ravel().reshape(-1, 1)

    if type(ps.dataset) not in [IMAGEDataset, PIXLDataset]:
        nav_img = ps.dataset.nav_img.data if ps.dataset.nav_img_bin is None else ps.dataset.nav_img_bin.data
        z_id = (nav_img / nav_img.max()).reshape(-1, 1)
    else:
        intensity_map = ps.dataset.intensity_map if ps.dataset.intensity_map_bin is None else ps.dataset.intensity_map_bin 
        z_id = (intensity_map / intensity_map.max()).reshape(-1, 1)

    combined = np.concatenate(
        [
            x_id,
            y_id,
            z_id,
            latent,
            dataset.reshape(-1, dataset.shape[-1]).round(2),
            labels.reshape(-1, 1),
        ],
        axis=1,
    )

    if ratio_to_be_shown != 1.0:
        sampled_combined = random.choices(
            combined, k=int(latent.shape[0] // (ratio_to_be_shown ** -1))
        )
        sampled_combined = np.array(sampled_combined)
    else:
        sampled_combined = combined

    source = pd.DataFrame(
        sampled_combined,
        columns=["x_id", "y_id", "z_id", "x", "y"] + feature_list + ["Cluster_id"],
        index=pd.RangeIndex(0, sampled_combined.shape[0], name="pixel"),
    )
    alt.data_transformers.disable_max_rows()

    # Brush
    brush = alt.selection(type="interval")
    interaction = alt.selection(
        type="interval",
        bind="scales",
        on="[mousedown[event.shiftKey], mouseup] > mousemove",
        translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
        zoom="wheel![event.shiftKey]",
    )

    # Points
    # Base layer: all points (dimmed, small)
    points_base = (
        alt.Chart(source)
        .mark_circle(size=3)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)),
            opacity=alt.value(0.2)
        )
    )

    # Highlight layer: selected points (larger, bright)
    points_selected = (
        alt.Chart(source)
        .mark_circle(size=20)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)),
            opacity=alt.condition(brush, alt.value(1.0), alt.value(0.0)),
            tooltip=[
                "Cluster_id:N",
                alt.Tooltip("x:Q", format=",.2f"),
                alt.Tooltip("y:Q", format=",.2f"),
            ],
        )
    )

    # Combine and add selections
    points = (
        (points_base + points_selected)
        .add_selection(brush, interaction)
        .properties(
            width=450,
            height=450,
            title=alt.TitleParams(text="Latent space")
        )
    )



    # Base chart for data tables
    ranked_text = alt.Chart(source).mark_bar().transform_filter(brush)

    # Data Bars
    columns = list()
    domain_barchart = (0, 1) if ps.dataset_norm.max() < 1.0 else (-4, 4)
    for item in feature_list:
        columns.append(
            ranked_text.encode(
                y=alt.Y(f"mean({item}):Q", scale=alt.Scale(domain=domain_barchart))
            ).properties(title=alt.TitleParams(text=item))
        )
    text = alt.hconcat(*columns)  # Combine bars

    # Heatmap
    if show_map == True:
        nav_img_df = pd.DataFrame(
            {"x_nav_img": x_id.ravel(), "y_nav_img": y_id.ravel(), "z_nav_img": z_id.ravel()}
        )
        nav_img = (
            alt.Chart(nav_img_df)
            .mark_square(size=6)
            .encode(
                x=alt.X("x_nav_img:O", axis=None),
                y=alt.Y("y_nav_img:O", axis=None),
                color=alt.Color(
                    "z_nav_img:Q", scale=alt.Scale(scheme="greys", domain=[1.0, 0.0])
                ),
            )
            .properties(width=ps.width*2, height=ps.height*2)
        )
        heatmap = (
            alt.Chart(source)
            .mark_square(size=5)
            .encode(
                x=alt.X("x_id:O", axis=None),
                y=alt.Y("y_id:O", axis=None),
                color=alt.Color(
                    "Cluster_id:N", scale=alt.Scale(domain=domain, range=range_)
                ),
                opacity=alt.condition(brush, alt.value(0.75), alt.value(0)),
            )
            .properties(width=ps.width*2, height=ps.height*2)
            .add_selection(brush)
        )
        heatmap_nav_img = nav_img + heatmap

    final_widgets = [points, heatmap_nav_img, text] if show_map == True else [points, text]

    # Build chart
    chart = (
        alt.hconcat(*final_widgets)
        .resolve_legend(color="independent")
        .configure_view(strokeWidth=0)
    )

    return chart


def show_cluster_distribution(ps: PixelSegmenter, **kwargs):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select_cluster = widgets.SelectMultiple(options=["All"] + cluster_options)
    plots_output = widgets.Output()

    all_fig = []
    with plots_output:
        for i in range(ps.n_components):
                fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
                all_fig.append(fig)

    def eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            if change.new == ("All",):
                for i in range(ps.n_components):
                        fig = ps.plot_single_cluster_distribution(cluster_num=i, **kwargs)
            else:
                for cluster in change.new:
                        fig = ps.plot_single_cluster_distribution(
                            cluster_num=int(cluster.split("_")[1]), **kwargs
                        )

    multi_select_cluster.observe(eventhandler, names="value")
    display(multi_select_cluster)
    save_fig(all_fig)
    display(plots_output)

#sigma2 improvement
#need a helper function to aid with plotting in view_phase_map
import re

def parse_rgb_string(s):
    """Convert 'rgb(r, g, b)' or 'rgba(r, g, b, a)' string into RGB tuple (0-1 range)."""
    match = re.match(r"rgba?\(([\d\s.,]+)\)", s)
    if match:
        parts = [float(x.strip()) for x in match.group(1).split(",")]
        if len(parts) >= 3:
            return tuple(p / 255.0 for p in parts[:3])
    raise ValueError(f"Invalid RGB string: {s}")



def view_phase_map(ps, alpha_cluster_map=0.6):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Get cluster labels excluding -1 (noise clusters)
    valid_labels = sorted([label for label in set(ps.labels) if label != -1])

    # Use ps.cluster_colors if available, fallback to default color_palette
    use_cluster_colors = hasattr(ps, "cluster_colors") and ps.cluster_colors
    if use_cluster_colors:
        # Clean colors: convert to hex if necessary
        cluster_color_list = []
        for label in valid_labels:
            raw_color = ps.cluster_colors.get(label, "#000000")
            try:
                rgb = mcolors.to_rgb(raw_color)
            except ValueError:
                rgb = parse_rgb_string(raw_color)
            cluster_color_list.append(mcolors.to_hex(rgb))

    else:
        # Fall back to default color palette using current cmap
        cmap = plt.get_cmap(ps.color_palette)
        cluster_color_list = [
            mcolors.to_hex(cmap(i / max(1, len(valid_labels) - 1)))
            for i in range(len(valid_labels))
        ]

    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = [
        widgets.ColorPicker(value=color, description=f"cluster_{label}", layout=layout_format)
        for label, color in zip(valid_labels, cluster_color_list)
    ]

    # Build initial color map
    cluster_color_rgb = []
    for c in cluster_color_list:
        try:
            cluster_color_rgb.append(mcolors.to_rgb(c))
        except ValueError:
            cluster_color_rgb.append(parse_rgb_string(c))

    # Add gray color for cluster -1 (noise)
    noise_color = "#999999"  # Gray color for noise
    cluster_color_rgb.append(mcolors.to_rgb(noise_color))  # Append gray to the colormap

    # Create the colormap (including noise color)
    cluster_color_map = ListedColormap(cluster_color_rgb)

    # Mask phase map for the noise cluster (-1) to make it transparent
    phase = ps.labels.reshape(ps.height, ps.width)
    phase_masked = phase.astype(float)
    phase_masked[phase == -1] = np.nan  # Replace -1 (noise) with NaN

    out = widgets.Output()
    with out:
        fig = ps.plot_phase_map(
            cmap=cluster_color_map,
            alpha_cluster_map=alpha_cluster_map,
            phase_override=phase_masked  # ✅ now passing masked phase map
        )
        plt.show()
        save_fig(fig)

    def change_color(_):
        out.clear_output()
        
        updated_colors = []
        for picker in color_pickers:
            try:
                updated_colors.append(mcolors.to_rgb(picker.value))
            except ValueError:
                updated_colors.append(parse_rgb_string(picker.value))
        newcmp = ListedColormap(updated_colors)

        # Update cluster_colors (so downstream uses are in sync)
        ps.cluster_colors = {label: mcolors.to_hex(rgb) for label, rgb in zip(valid_labels, updated_colors)}
        ps.set_color_palette(newcmp)  # If you want to update internal palette too

        with out:
            fig = ps.plot_phase_map(cmap=newcmp, alpha_cluster_map=alpha_cluster_map)
            plt.show()
            save_fig(fig)

    button = widgets.Button(description="Set", layout=Layout(width="auto"))
    button.on_click(change_color)

    color_rows = [widgets.HBox(color_pickers[i:i+5]) for i in range(0, len(color_pickers), 5)]
    color_box = widgets.VBox([*color_rows, button], layout=Layout(flex="2 1 0%", width="auto"))
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))

    final_box = widgets.VBox([color_box, out_box])
    display(final_box)

def show_unmixed_weights(weights: pd.DataFrame):
    weights_options = weights.index
    multi_select_cluster = widgets.SelectMultiple(options=weights_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()

    with all_output:
        display(weights)

    def multi_select_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            row_index = [cluster for cluster in change.new]
            display(weights.loc[row_index])
            for cluster in change.new:
                num_cpnt = len(weights.columns.to_list())
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                axs.bar(
                    np.arange(0, num_cpnt),
                    weights[weights.index == cluster].to_numpy().ravel(),
                    width=0.6,
                )
                axs.set_xticks(np.arange(0, num_cpnt))
                axs.set_ylabel("weight of component")
                axs.set_xlabel("component number")
                plt.show()

    multi_select_cluster.observe(multi_select_cluster_eventhandler, names="value")

    display(multi_select_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, "All weights")
    tab.set_title(1, "Single weight")
    display(tab)


def show_unmixed_components(ps: PixelSegmenter, components: pd.DataFrame):
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output = widgets.Output()
    all_output = widgets.Output()

    with all_output:
        ps.plot_unmixed_profile(components)

    def dropdown_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            visual.plot_profile(ps.energy_axis, components[change.new], ps.peak_list)

    dropdown_cluster.observe(dropdown_cluster_eventhandler, names="value")

    display(dropdown_cluster)
    tab = widgets.Tab([all_output, plots_output])
    tab.set_title(0, "All cpnt")
    tab.set_title(1, "Single cpnt")
    display(tab)


def show_unmixed_weights_and_compoments(
    ps: PixelSegmenter, weights: pd.DataFrame, components: pd.DataFrame
):
    # weights
    # weights.loc["Sum"] = weights.sum()
    # weights = weights.round(3)
    # weights_options = weights.index
    multi_select_cluster = widgets.SelectMultiple(options=weights.index)
    plots_output = widgets.Output()
    all_output = widgets.Output()

    with all_output:
        display(weights)

    def multi_select_cluster_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            row_index = [cluster for cluster in change.new]
            display(weights.loc[row_index])
            for cluster in change.new:
                num_cpnt = len(weights.columns.to_list())
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                axs.bar(
                    np.arange(0, num_cpnt),
                    weights[weights.index == cluster].to_numpy().ravel(),
                    width=0.6,
                )
                axs.set_xticks(np.arange(0, num_cpnt))
                axs.set_ylabel("Abundance coefficient")
                axs.set_xlabel("NMF component ID")
                plt.show()
                save_fig(fig)

    multi_select_cluster.observe(multi_select_cluster_eventhandler, names="value")

    # compoments
    components_options = components.columns
    dropdown_cluster = widgets.Dropdown(options=components_options)
    plots_output_cpnt = widgets.Output()
    all_output_cpnt = widgets.Output()

    with all_output_cpnt:
        fig = ps.plot_unmixed_profile(components)
        save_fig(fig)

    def dropdown_cluster_eventhandler(change):
        plots_output_cpnt.clear_output()
        with plots_output_cpnt:
            if type(ps.dataset) in [IMAGEDataset, PIXLDataset]:
                fig, axs = plt.subplots(1,1)
                axs.bar(
                    ps.dataset.feature_list,
                    components[change.new],
                    width=0.6,
                    linewidth=1,
                )
                for i in range(len(ps.dataset.feature_list)):
                    y = components[change.new][i] + components[change.new].max()*0.03
                    axs.text(i-len(ps.dataset.feature_list[i])*0.11,y,ps.dataset.feature_list[i], fontsize=8)
                    
                axs.set_ylim(None, components[change.new].max()*1.2)
                axs.set_xticks([])
                axs.set_xticklabels([])
                axs.set_title(f"{change.new}")
                plt.show()
            else:
                visual.plot_profile(ps.energy_axis, components[change.new], ps.peak_list)

    dropdown_cluster.observe(dropdown_cluster_eventhandler, names="value")

    widget_set = widgets.HBox([multi_select_cluster, dropdown_cluster])
    display(widget_set)

    tab = widgets.Tab([all_output, plots_output, all_output_cpnt, plots_output_cpnt])
    tab.set_title(0, "All weights")
    tab.set_title(1, "Single weight")
    tab.set_title(2, "All components")
    tab.set_title(3, "Single component")
    display(tab)


def view_clusters_sum_spectra(
    ps: PixelSegmenter, normalisation=True, spectra_range=(0, 8)
):
    cluster_options = [f"cluster_{n}" for n in range(ps.n_components)]
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()
    profile_output = widgets.Output()

    figs = []
    with plots_output:
        for cluster in cluster_options:
            fig = ps.plot_binary_map_spectra_profile(
                cluster_num=int(cluster.split("_")[1]),
                normalisation=normalisation,
                spectra_range=spectra_range,
            )
            figs.append(fig)

    def eventhandler(change):
        plots_output.clear_output()
        profile_output.clear_output()

        with plots_output:
            for cluster in change.new:
                fig = ps.plot_binary_map_spectra_profile(
                    cluster_num=int(cluster.split("_")[1]),
                    normalisation=normalisation,
                    spectra_range=spectra_range,
                )

        with profile_output:
            ### X-ray profile ###
            for cluster in change.new:
                _, _, spectra_profile = ps.get_binary_map_spectra_profile(
                    cluster_num=int(cluster.split("_")[1]), use_label=True
                )
                visual.plot_profile(
                    spectra_profile["energy"], spectra_profile["intensity"], ps.peak_list
                )

    multi_select.observe(eventhandler, names="value")

    display(multi_select)
    save_fig(figs)
    tab = widgets.Tab([plots_output, profile_output])
    tab.set_title(0, "clusters + spectra")
    tab.set_title(1, "spectra")
    display(tab)


def save_csv(df):
    text = widgets.Text(
        value="file_name.csv",
        placeholder="Type something",
        description="Save as:",
        disabled=False,
        continuous_update=True,
    )

    button = widgets.Button(description="Save")
    out = widgets.Output()

    def save_to(_):
        out.clear_output()
        with out:
            df.to_csv(text.value)
            print("save the csv to", text.value)

    button.on_click(save_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)


def show_cluster_stats(ps: PixelSegmenter, binary_filter_args={}):
    columns = [
        "area (um^2)",
        "equivalent_diameter (um)",
        "major_axis_length (um)",
        "minor_axis_length (um)",
    ]

    for item in ("min_intensity", "mean_intensity", "max_intensity"):
        columns += [f"{item}_{peak}" for peak in ps.peak_list]

    properties = widgets.Dropdown(options=columns, description="property:")
    clusters = widgets.SelectMultiple(
        options=[f"cluster_{i}" for i in range(ps.n_components)], description="cluster:"
    )
    bound_bins = widgets.BoundedIntText(
        value=40, min=5, max=100, step=1, description="num_bins:"
    )
    output = widgets.Output()

    def plot_output(clusters, properties, bound_bins):
        output.clear_output()
        df_list = []
        fig_list = []
        with output:
            for cluster in clusters:
                df_stats = ps.phase_statics(
                    cluster_num=int(cluster.split("_")[1]),
                    element_peaks=ps.peak_list,
                    binary_filter_args=binary_filter_args,
                )
                df_list.append(df_stats[properties])
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                sns.set_style("ticks")
                sns.histplot(df_stats[properties], bins=bound_bins)
                plt.title(cluster)
                plt.show()
                fig_list.append(fig)
            df_list = pd.concat(df_list, axis=1, keys=clusters)

        save_csv(df_list)
        fig_list = fig_list[0] if len(fig_list) == 1 else fig_list
        save_fig(fig_list)

    def cluster_handler(change):
        plot_output(change.new, properties.value, bound_bins.value)

    def properties_handler(change):
        plot_output(clusters.value, change.new, bound_bins.value)

    def bins_handler(change):
        plot_output(clusters.value, properties.value, change.new)

    clusters.observe(cluster_handler, names="value")
    properties.observe(properties_handler, names="value")
    bound_bins.observe(bins_handler, names="value")

    all_widgets = widgets.HBox([clusters, properties, bound_bins])
    display(all_widgets)
    display(output)


def view_emi_dataset(tem, search_energy=True):
    if search_energy == True:
        search_energy_peak()

    nav_img_out = widgets.Output()
    with nav_img_out:
        tem.nav_img.plot(colorbar=False)
        plt.show()
        fig, axs = plt.subplots(1, 1)
        axs.imshow(tem.nav_img.data, cmap="gray")
        axs.axis("off")
        save_fig(fig)
        plt.close()

    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(tem.spectra)

    elemental_map_out = widgets.Output()
    with elemental_map_out:
        if len(tem.feature_list) != 0:
            pick_color(
                visual.plot_intensity_maps, spectra=tem.spectra, element_list=tem.feature_list
            )
            # fig = visual.plot_intensity_maps(sem.spectra, sem.feature_list)
            # save_fig(fig)

    if tem.spectra_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            pick_color(
                visual.plot_intensity_maps,
                spectra=tem.spectra_bin,
                element_list=tem.feature_list,
            )
            # fig = visual.plot_intensity_maps(sem.spectra_bin, sem.feature_list)
            # save_fig(fig)

    default_elements = ""
    for i, element in enumerate(tem.feature_list):
        if i == len(tem.feature_list) - 1:
            default_elements += element
        else:
            default_elements += element + ", "

    layout = widgets.Layout(width="400px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        disabled=False,
        continuous_update=True,
        # display='flex',
        # flex_flow='column',
        align_items="stretch",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            tem.set_feature_list(feature_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(tem.spectra)

        if len(tem.feature_list) != 0:
            elemental_map_out.clear_output()
            with elemental_map_out:
                pick_color(
                    visual.plot_intensity_maps,
                    spectra=tem.spectra,
                    element_list=tem.feature_list,
                )
                visual.plot_intensity_maps(tem.spectra, tem.feature_list)

        if tem.spectra_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(tem.spectra_bin, tem.feature_list)

    button.on_click(set_to)
    all_widgets = widgets.HBox([text, button])
    display(all_widgets)
    display(out)

    tab_list = [nav_img_out, sum_spec_out, elemental_map_out]
    if tem.spectra_bin is not None:
        tab_list += [elemental_map_out_bin]

    tab = widgets.Tab(tab_list)
    tab.set_title(0, "Sum intensity map")
    tab.set_title(1, "Sum spectrum")
    tab.set_title(2, "Elemental maps (raw)")
    i = 2
    if tem.spectra_bin is not None:
        tab.set_title(i + 1, "Elemental maps (binned)")
    display(tab)


def show_abundance_map(ps:PixelSegmenter, weights:pd.DataFrame, components: pd.DataFrame):
    def plot_rgb(ps, phases:List):
        shape = ps.get_binary_map_spectra_profile(0)[0].shape
        img = np.zeros((shape[0], shape[1], 3))
        
        # make abundance map
        for i, phase in enumerate(phases):
            if phase!='None':
                cpnt_weights = weights[phase]/weights[phase].max()
                tmp = np.zeros(shape)
                for j in range(ps.n_components):
                    try:
                        idx = ps.get_binary_map_spectra_profile(j)[1]
                        tmp[idx] = cpnt_weights[j]
                    except ValueError:
                        pass
                        # print(f'warning: no pixel is assigned to cpnt_{j}.')
                img[:, :, i] = tmp
            else:
                img[:, :, i] = np.zeros(shape)

        fig, axs = plt.subplots(1, 1, dpi=96)
        axs.imshow(img, alpha=0.95)
        axs.axis("off")
        plt.show()
        return fig
    
    cpnt_names = [f'cpnt_{i}' for i in range(len(weights.columns))] 
    cpnt_options = [x for x in zip(cpnt_names, weights.columns)] + [('None', 'None')]
    dropdown_r = widgets.Dropdown(options=cpnt_options, value='None', description="Red:")
    dropdown_g = widgets.Dropdown(options=cpnt_options, value='None', description="Green:")
    dropdown_b = widgets.Dropdown(options=cpnt_options, value='None', description="Blue:")

    plots_output = widgets.Output()
    with plots_output:
        fig = plot_rgb(
            ps,
            phases=['None', 'None', 'None'],
        )
        save_fig(fig)

    def dropdown_r_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[change.new, dropdown_g.value, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_g_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[dropdown_r.value, change.new, dropdown_b.value],
            )
            save_fig(fig)

    def dropdown_b_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(
                ps,
                phases=[dropdown_r.value, dropdown_g.value, change.new],
            )
            save_fig(fig)

    dropdown_r.observe(dropdown_r_eventhandler, names="value")
    dropdown_g.observe(dropdown_g_eventhandler, names="value")
    dropdown_b.observe(dropdown_b_eventhandler, names="value")
    color_box = widgets.VBox([dropdown_r, dropdown_g, dropdown_b])

    display(color_box)
    display(plots_output)

    
def plot_ternary_composition(ps:PixelSegmenter):
    # cluster_num:int,
    # elements:List,
    # k_factors:List[float]=None,
    # composition_units:str='atomic',
    cluster_options = range(ps.n_components)
    dropdown_cluster = widgets.Dropdown(options=cluster_options, description='cluster',value=None, )
    
    element_options = []
    for el in ps.dataset.feature_list:
        if el in k_factors_120kV.keys():
            element_options.append(el)
    
    dropdown_element1 = widgets.Dropdown(options=element_options, description='element1',value=None)
    dropdown_element2 = widgets.Dropdown(options=element_options, description='element2',value=None)
    dropdown_element3 = widgets.Dropdown(options=element_options, description='element3',value=None)
    
    plots_output = widgets.Output()
    
    button = widgets.Button(description="Calculate")
    out = widgets.Output()

    def button_evenhandler(_):
        plots_output.clear_output()
        with plots_output:
            try:
                ps.plot_ternary_composition(
                    cluster_num=int(dropdown_cluster.value),
                    elements=[dropdown_element1.value, dropdown_element2.value, dropdown_element3.value],
                )
            except ValueError:
                print('Oops. Please try different combiniations of elements')
            except TypeError:
                print('Please select cluster number')
            except AttributeError:
                print('Please select elements')

    button.on_click(button_evenhandler)
    
    option_box = widgets.HBox([dropdown_cluster, 
                               dropdown_element1, 
                               dropdown_element2,
                               dropdown_element3, 
                               button],
                              layout=Layout(flex="flex-start", width="80%",align_items='center'),)
    display(option_box)
    display(plots_output)


    ##########################
    #         SIGMA2         #
    ##########################




from plotly.subplots import make_subplots
from ipywidgets import Button, Output, ToggleButtons, Dropdown, HBox, VBox, Layout
from IPython.display import clear_output


from collections import defaultdict






def interactive_latent_plot(ps, ratio_to_be_shown=0.5,n_colours=30):
    """
    Interactive GUI for merging the colours of the clusters in the latent space plot.
    
    Start by clicking the cluster(s) you wish to be rendered the samae colour.
    
    Also allows clusters to be selected using lasso/rectangle tool
    
    Pressing the 'Merge Clusters' button will re-render the plot, with these clusters being the same colour.
    
    Parameters
    -----------
    ps :                PixelSegmenter object
                        The PixelSegmenter object to be visualised.
         
    ratio_to_be_shown : float
                        The proportion of points to be shown. Default = 0.5, so half the points. 
                        Can improve rendering times by reducing this value.
                        
    n_colours :         int
                        Number of colours to display that the user may choose from
    
    
    
    """
    
    manual_cluster_colors = {}  # cluster_id -> RGB string
    original_palette='%s' % ps.color_palette #copying the original color palette
    
    original_labels = ps.labels.copy()

    
    #for creating new clusters
    selected_point_indices = set()  # for new cluster selection
    new_cluster_mode = [False]  # wrapped in list for mutability
    
    original_cluster_colors = {
        i: "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for i, (r, g, b, _) in enumerate(
            cm.get_cmap(original_palette)(np.linspace(0, 1, ps.n_components))
        )
    } #getting the original cluster colors object.

    
    def get_color(cid):
        if cid == -1:
            return '#999999'  # Dim gray for noise
        return (
            manual_cluster_colors.get(cid)
            or ps.cluster_colors.get(cid)
            or original_cluster_colors.get(cid, '#cccccc')
        )



    # Get the colormap and normalization
    cmap = cm.get_cmap(ps.color_palette)
    norm = ps.color_norm



    # Extract relevant data
    latent, dataset, feature_list, labels = (
        ps.latent,
        ps.dataset.normalised_elemental_data,
        ps.dataset.feature_list,
        ps.labels,
    )

    # Subsample if requested
    if ratio_to_be_shown != 1.0:
        sampled_indices = random.sample(range(latent.shape[0]), int(latent.shape[0] * ratio_to_be_shown))
    else:
        sampled_indices = list(range(latent.shape[0]))

    df = pd.DataFrame(latent[sampled_indices], columns=["x", "y"])
    df["cluster"] = labels.flatten()[sampled_indices]
    df["index"] = sampled_indices

    # State tracking
    selected_clusters = set()
    merged_clusters = defaultdict(list)
    out = Output()
    new_ps = None  # Will hold the new PixelSegmenter if created

    # Build initial figure widget
    fig = go.FigureWidget()
    fig.update_layout(
        title="Interactive Latent Space",
        xaxis_title="Latent X",
        yaxis_title="Latent Y",
        plot_bgcolor='white',
        height=600,
        width=600,
        dragmode="select",  # Default to select
        selectdirection="any"
    )
    
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey',
    zerolinecolor='lightgrey'
    )
    fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey',
    zerolinecolor='lightgrey'
    )


    # Helper to regenerate the plot with merged colors
    def plot():
        nonlocal df
        # Refresh the DataFrame with updated labels
        current_labels = ps.labels.flatten()
        df = pd.DataFrame(latent[sampled_indices], columns=["x", "y"])
        df["cluster"] = current_labels[sampled_indices]
        df["index"] = sampled_indices

        with fig.batch_update():
            fig.data = []  # Clear previous data

            # Remap clusters based on merged state
            cluster_map = {}
            for original_cluster in sorted(df['cluster'].unique()):
                for group_id, originals in merged_clusters.items():
                    if original_cluster in originals:
                        cluster_map[original_cluster] = group_id
                        break
                else:
                    cluster_map[original_cluster] = original_cluster

            merged_ids = df['cluster'].map(cluster_map)
            colors = merged_ids.map(lambda cid: manual_cluster_colors.get(cid, get_color(cid)))

            fig.add_trace(go.Scattergl(
                x=df['x'],
                y=df['y'],
                mode='markers',
                marker=dict(size=6, color=colors),
                name='Clusters',
                customdata=np.stack([df['cluster'], df['index']], axis=1),
                hovertemplate="Cluster: %{customdata[0]}<br>Index: %{customdata[1]}<extra></extra>"
            ))

            trace = fig.data[0]
            trace.on_click(on_point_click)
            trace.on_selection(on_select)
            
            
    # Click handler
    def on_point_click(trace, points, selector):
        for i in points.point_inds:
            cluster_id = int(trace.customdata[i][0])

            # Skip noise (cluster -1)
            if cluster_id == -1:
                continue  # Ignore noise cluster (-1)

            # Add valid cluster IDs to the selected_clusters set
            selected_clusters.add(cluster_id)

        # Print out the selected clusters, excluding noise
        with out:
            print(f"Clicked cluster(s): {sorted(selected_clusters)}")

    
    #toggle switch to handle unassigned points when creating new cluster
    include_noise_toggle = ToggleButtons(
        options=["Include -1", "Exclude -1"],
        description="Noise pts:",
        value="Exclude -1",
        style={"button_width": "100px"}
    )

    # Merge button
    merge_button = Button(description="Merge Clusters")
    
    
    #new cluster buttons
    select_points_button = Button(description="Select Points for New Cluster")
    create_cluster_button = Button(description="Create New Cluster")
    
    # Selection handler (lasso or rectangle)
    def on_select(trace, points, selector):
        if not points.point_inds:
            return

        if new_cluster_mode[0]:
            indices = [int(trace.customdata[i][1]) for i in points.point_inds]
            selected_point_indices.update(indices)
            with out:
                print(f"🟢 Selected {len(indices)} points for new cluster.")
        else:
            selected = set(int(trace.customdata[i][0]) for i in points.point_inds)
            selected_clusters.update(selected)
            with out:
                print(f"Lasso/Box selected clusters: {sorted(selected_clusters)}")
     
    def on_select_points_clicked(b):
        selected_point_indices.clear()
        new_cluster_mode[0] = True
        with out:
            out.clear_output()
            print("🎯 Selection mode activated. Use lasso/box to select points, then click 'Create New Cluster'.")

    def on_create_cluster_clicked(b):
        if not selected_point_indices:
            with out:
                print("⚠️ No points selected.")
            return

        nonlocal labels
        current_labels = labels.flatten()
        existing_clusters = set(current_labels)
        new_cluster_id = max(existing_clusters) + 1

        # Store selected points count before clearing
        num_points = len(selected_point_indices)

        # Assign new cluster ID
        for idx in selected_point_indices:
            if include_noise_toggle.value == "Include -1" or current_labels[idx] != -1:
                current_labels[idx] = new_cluster_id


        labels = current_labels.reshape(ps.height * ps.width, 1)
        ps.labels = labels
        ps.n_components = len(set(labels.flatten()))

        # Reset selection state
        selected_point_indices.clear()
        new_cluster_mode[0] = False

        with out:
            out.clear_output()
            print(f"✅ Created new cluster {new_cluster_id} with {num_points} points.")

        plot()




    

    def on_merge_clicked(b):
        if not selected_clusters:
            with out:
                print("No clusters selected to merge.")
            return

        # Skip noise cluster (-1) from selection
        selected_clusters = {cluster for cluster in selected_clusters if cluster != -1}

        if not selected_clusters:  # If no clusters are left after excluding noise
            with out:
                print("No valid clusters selected (all were noise clusters).")
            return

        resolved = set()
        for cluster in selected_clusters:
            for group_id, originals in merged_clusters.items():
                if cluster in originals:
                    resolved.add(group_id)
                    break
            else:
                resolved.add(cluster)

        if len(resolved) <= 1:
            with out:
                print("Nothing to merge (already in same group?).")
            return

        new_id = min(resolved)
        new_group = set()
        for group in resolved:
            if group in merged_clusters:
                new_group.update(merged_clusters.pop(group))
            else:
                new_group.add(group)
        new_group.add(new_id)
        merged_clusters[new_id] = list(sorted(new_group))

        selected_clusters.clear()
        with out:
            out.clear_output()
            print(f"Merged clusters {sorted(resolved)} into cluster {new_id}")
        plot()


    # Reset button
    reset_button = Button(description="Reset")




    def on_reset_clicked(b):
        confirm_out.clear_output()
        
        with confirm_out:
            print("⚠️ Are you sure you want to reset everything?")
            confirm = Button(description="Yes, Reset", button_style='danger')
            cancel = Button(description="Cancel", button_style='info')
            
            def do_reset(btn):
                nonlocal new_ps, labels

                selected_clusters.clear()
                merged_clusters.clear()
                manual_cluster_colors.clear()
                new_ps = None

                ps.color_palette = original_palette
                ps.cluster_colors = original_cluster_colors.copy()

                # Reset labels
                ps.labels = original_labels.copy()
                labels = ps.labels  # local reference
                ps.n_components = len(set(labels.flatten()))

                fig.data = []
                confirm_out.clear_output()
                out.clear_output()
                print("✅ Full reset complete. All cluster assignments and colors reverted.")

                plot()

            def cancel_reset(btn):
                confirm_out.clear_output()
                with out:
                    out.clear_output()
                    print("❎ Reset cancelled.")

            confirm.on_click(do_reset)
            cancel.on_click(cancel_reset)

            display(HBox([confirm, cancel]))


    # Output button
    output_button = Button(description="Output Merged Clusters")

    def on_output_clicked(b):
        nonlocal new_ps
        with out:
            print('Clicked output')

        if not merged_clusters:
            with out:
                print("No merged clusters to output.")
            return
        else:
            with out:
                print("Merging Clusters...")
                new_labels = labels.copy()
                for merged_id, originals in merged_clusters.items():
                    for original in originals:
                        new_labels[labels == original] = merged_id
        
                # Create a new PixelSegmenter instance
                new_ps = PixelSegmenter(
                    dataset=ps.dataset,
                    latent=ps.latent)
                new_ps.labels=new_labels
                new_ps.n_components=len(set(new_labels.flatten()))
                
                print("✅ New PixelSegmenter object created!")
                
    # Color picker options (define however many you want)
    # Build a color selector UI

    # Generate colors from the colormap
    color_buttons = []
    num_colors = n_colours
    cmap_colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

    selected_color = [None]  # use list for mutability in closure

    def on_color_button_click(b):
        selected_color[0] = b.tooltip  # stores the RGB string
        with out:
            print(f"Selected color: {selected_color[0]}")

    for rgba in cmap_colors:
        r, g, b_, _ = rgba
        rgb_str = f'rgb({int(r*255)}, {int(g*255)}, {int(b_*255)})'
        btn = Button(
            layout=Layout(width='30px', height='30px'),
            tooltip=rgb_str,
            style={'button_color': rgb_str},
        )
        btn.on_click(on_color_button_click)
        color_buttons.append(btn)

    color_selector_ui = HBox(color_buttons[:15]), HBox(color_buttons[15:])


    recolor_button = Button(description="Recolour Selected Clusters")

    def on_recolor_clicked(b):
        if not selected_clusters or not selected_color[0]:
            with out:
                print("Please select clusters and a color first.")
            return

        for cluster_id in selected_clusters:
            if cluster_id == -1:
                continue  # Skip noise cluster (-1)
            
            # Recolor the valid clusters
            manual_cluster_colors[cluster_id] = selected_color[0]
            ps.cluster_colors[cluster_id] = selected_color[0]
            
        selected_clusters.clear()  # Clear selection to avoid recoloring again
        with out:
            out.clear_output()
            print("✅ Recolored selected clusters.")
        plot()
        
    clear_selection_button = Button(description="Clear Selection")
        
    def on_clear_selection_clicked(b):
        selected_clusters.clear()
        with out:
            out.clear_output()
            print("🧼 Cleared selected clusters.")


    
    confirm_out = Output()



    # Bind buttons
    merge_button.on_click(on_merge_clicked)
    reset_button.on_click(on_reset_clicked)
    output_button.on_click(on_output_clicked)
    recolor_button.on_click(on_recolor_clicked)
    select_points_button.on_click(on_select_points_clicked)
    create_cluster_button.on_click(on_create_cluster_clicked)
    clear_selection_button.on_click(on_clear_selection_clicked)


    # Initial plot
    plot()

    # Layout
    controls = HBox([
        reset_button,
        clear_selection_button,
        merge_button, 
        output_button, 
        recolor_button,
        select_points_button,
        create_cluster_button,
        include_noise_toggle
    ])

    display(VBox([controls, *color_selector_ui, fig, confirm_out, out]))


    # Return access function for new object
    return lambda: new_ps
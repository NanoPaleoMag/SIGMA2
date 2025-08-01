#doing relative imports
import sys # for relatove imports of sigma
sys.path.insert(0,"../..")



from sigma.utils import visualisation as visual
from sigma.src.segmentation import PixelSegmenter
from sigma.utils.load import SEMDataset, IMAGEDataset, PIXLDataset
from sigma.utils.loadtem import TEMDataset
from sigma.src.utils import k_factors_120kV
from sigma.utils import normalisation as norm

from sklearn.preprocessing import RobustScaler
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


import matplotlib.colors as mcolors

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


def view_dataset(dataset: Union[SEMDataset, TEMDataset, IMAGEDataset], search_energy=True):
    """
    GUI for visualisation of the dataset.
    """

    if search_energy:
        search_energy_peak()

    # --- Navigation Image ---
    nav_img_out = widgets.Output()
    with nav_img_out:
        if dataset.nav_img is not None:
            dataset.nav_img.plot(colorbar=False)
            plt.show()
            fig, axs = plt.subplots(1, 1)
            axs.imshow(dataset.nav_img.data, cmap="gray")
            axs.axis("off")
            save_fig(fig)
            plt.close()

    # --- Sum Spectrum ---
    sum_spec_out = widgets.Output()
    with sum_spec_out:
        visual.plot_sum_spectrum(dataset.spectra)

    # --- Elemental Maps (Raw) ---
    elemental_map_out = widgets.Output()
    with elemental_map_out:
        nav_img = dataset.nav_img_feature if dataset.nav_img_feature is not None else None
        pick_color(
            visual.plot_intensity_maps,
            spectra=dataset.spectra,
            element_list=dataset.feature_list,
            include_nav_img=nav_img,
        )

    # --- Elemental Maps (Binned) ---
    elemental_map_out_bin = None
    if dataset.spectra_bin is not None:
        elemental_map_out_bin = widgets.Output()
        with elemental_map_out_bin:
            nav_img_bin = (
                dataset.nav_img_feature.rebin(dataset.spectra_bin.axes_manager.navigation_shape)
                if dataset.nav_img_feature is not None
                else None
            )
            pick_color(
                visual.plot_intensity_maps,
                spectra=dataset.spectra_bin,
                element_list=dataset.feature_list,
                include_nav_img=nav_img_bin,
            )

    # --- Feature List Editor ---
    default_elements = ", ".join(dataset.feature_list)

    layout = widgets.Layout(width="400px", height="40px")
    text = widgets.Text(
        value=default_elements,
        placeholder="Type something",
        description="Feature list:",
        layout=layout,
    )

    button = widgets.Button(description="Set")
    out = widgets.Output()

    def set_to(_):
        out.clear_output()
        with out:
            feature_list = text.value.replace(" ", "").split(",")
            dataset.set_feature_list(feature_list)

        # Refresh sum spectrum
        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(dataset.spectra)

        # Refresh raw elemental maps
        elemental_map_out.clear_output()
        with elemental_map_out:
            nav_img = dataset.nav_img_feature if dataset.nav_img_feature is not None else None
            visual.plot_intensity_maps(dataset.spectra, dataset.feature_list, include_nav_img=nav_img)

        # Refresh binned elemental maps
        if dataset.spectra_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                nav_img_bin = (
                    dataset.nav_img_feature.rebin(dataset.spectra_bin.axes_manager.navigation_shape)
                    if dataset.nav_img_feature is not None
                    else None
                )
                visual.plot_intensity_maps(dataset.spectra_bin, dataset.feature_list, include_nav_img=nav_img_bin)

    button.on_click(set_to)
    display(widgets.HBox([text, button]))
    display(out)

    # --- Tabs ---
    tab_list = [nav_img_out, sum_spec_out, elemental_map_out]
    tab_titles = ["Navigation Signal", "Sum spectrum", "Elemental maps (raw)"]

    if dataset.spectra_bin is not None:
        tab_list.append(elemental_map_out_bin)
        tab_titles.append("Elemental maps (binned)")

    tab = widgets.Tab(tab_list)
    for i, title in enumerate(tab_titles):
        tab.set_title(i, title)
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

            # Deduplicate while preserving order
            seen = set()
            cleaned_list = []
            for el in feature_list:
                if el not in seen:
                    cleaned_list.append(el)
                    seen.add(el)

            # Ensure "Navigator" appears once at the end if it's anywhere in the list
            if "Navigator" in cleaned_list:
                cleaned_list = [el for el in cleaned_list if el != "Navigator"]
                cleaned_list.append("Navigator")

            dataset.set_feature_list(cleaned_list)

        sum_spec_out.clear_output()
        with sum_spec_out:
            visual.plot_sum_spectrum(dataset.spectra)

        elemental_map_out.clear_output()
        with elemental_map_out:
            visual.plot_intensity_maps(dataset.spectra, dataset.feature_list, include_nav_img=dataset.nav_img_feature)

        if dataset.spectra_bin is not None:
            elemental_map_out_bin.clear_output()
            with elemental_map_out_bin:
                visual.plot_intensity_maps(dataset.spectra_bin, dataset.feature_list)



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


def check_latent_space(ps: PixelSegmenter, ratio_to_be_shown=0.25, show_map=False,alpha_cluster_map=0.75):
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


    latent = ps.latent
    raw_dataset = ps.dataset.normalised_elemental_data
    feature_list = ps.dataset.feature_list
    labels = ps.labels

    # Apply robust scaling to the elemental features
    scaler = RobustScaler()
    robust_scaled_dataset = scaler.fit_transform(raw_dataset.reshape(-1, raw_dataset.shape[-1]))

    # Reshape back if needed
    dataset = robust_scaled_dataset
    x_id, y_id = np.meshgrid(range(ps.width), range(ps.height))
    x_id = x_id.ravel().reshape(-1, 1)
    y_id = y_id.ravel().reshape(-1, 1)

    if type(ps.dataset) not in [IMAGEDataset, PIXLDataset]:
        nav_img = ps.dataset.nav_img_bin if ps.dataset.nav_img_bin is not None else ps.dataset.nav_img

        # Resize nav_img to match spectra dimensions
        from skimage.transform import resize
        target_shape = ps.dataset.spectra_bin.data.shape[:2] if ps.dataset.spectra_bin is not None else ps.dataset.spectra.data.shape[:2]
        nav_img_resized = resize(nav_img.data, target_shape, preserve_range=True, anti_aliasing=True)

        z_id = (nav_img_resized / nav_img_resized.max()).reshape(-1, 1)
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



    ranked_text = alt.Chart(source).mark_bar(clip=True).transform_filter(brush)

    # Data Bars
    columns = list()
    domain_barchart = (0, 1) if ps.dataset_norm.max() < 1.0 else (-4, 4)



    for item in feature_list:
        chart = (
            ranked_text.encode(
                y=alt.Y(
                    f"mean({item}):Q",
                    title=item,
                    scale=alt.Scale(domain=domain_barchart),
                    axis=alt.Axis(titleFontSize=10)
                )
            )
            .properties(
                title=alt.TitleParams(
                    text=item,
                    anchor="middle",
                    fontSize=11,
                    dy=-10  # Pull the title closer to chart (optional tweak)
                )
            )
        )
        columns.append(chart)

    text = alt.hconcat(*columns)


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
                opacity=alt.condition(brush, alt.value(alpha_cluster_map), alt.value(0)),
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
    from IPython.display import display
    import ipywidgets as widgets
    import matplotlib.colors as mcolors

    # Utility: Get current non-empty cluster labels
    def get_current_clusters():
        current_labels = ps.labels
        unique_labels = sorted(set(current_labels.flatten()))
        return [c for c in unique_labels if c >= 0 and np.sum(current_labels == c) > 0]

    # Utility: Get RGB color for a cluster
    def get_cluster_color(cluster_id):
        default_color = "#000000"
        raw_color = ps.cluster_colors.get(cluster_id, default_color)
        try:
            return mcolors.to_rgb(raw_color)
        except ValueError:
            return parse_rgb_string(raw_color)

    # Recompute stats (mu, label shape etc.)
    refresh_cluster_statistics(ps)

    # Prepare cluster options
    current_clusters = get_current_clusters()
    cluster_options = [f"cluster_{n}" for n in current_clusters]

    # Widgets
    cluster_selector = widgets.SelectMultiple(
        options=["All"] + cluster_options,
        value=("All",),
        description="Clusters:"
    )

    format_selector = widgets.Dropdown(
        options=["png", "svg", "pdf"],
        value="png",
        description="Save as:"
    )

    save_button = widgets.Button(description="💾 Save Plot")

    plots_output = widgets.Output()
    control_bar = widgets.HBox([cluster_selector, format_selector, save_button])

    # Helper: Save figure list to files
    def save_fig(figs, fmt):
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"cluster_plots_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)

        for i, fig in enumerate(figs):
            path = os.path.join(out_dir, f"cluster_{i}.{fmt}")
            fig.savefig(path, format=fmt, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {out_dir}")

    # Main plotting function
    def plot_clusters(cluster_indices):
        plots_output.clear_output(wait=True)
        all_figs = []

        with plots_output:
            for i in cluster_indices:
                try:
                    fig = ps.plot_single_cluster_distribution(
                        cluster_num=i,
                        color=get_cluster_color(i),  # 🎨 pass color here
                        **kwargs
                    )
                    all_figs.append(fig)
                except Exception as e:
                    print(f"Skipping cluster {i} due to error: {e}")
        return all_figs

    # Event: Cluster selection changed
    def on_cluster_selection(change):
        selected = change.new
        if selected == ("All",):
            selected_indices = get_current_clusters()
        else:
            selected_indices = [int(c.split("_")[1]) for c in selected]
        nonlocal current_figs
        current_figs = plot_clusters(selected_indices)

    # Event: Save button clicked
    def on_save_clicked(_):
        if current_figs:
            save_fig(current_figs, format_selector.value)

    # Attach handlers
    cluster_selector.observe(on_cluster_selection, names="value")
    save_button.on_click(on_save_clicked)

    # Display everything
    display(control_bar)
    display(plots_output)

    # Initial plot
    current_figs = plot_clusters(current_clusters)

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

    def build_color_map():
        labels = ps.labels.flatten()
        valid_labels = sorted(set(labels) - {-1})
        cluster_colors = {}

        use_cluster_colors = hasattr(ps, "cluster_colors") and ps.cluster_colors
        if use_cluster_colors:
            for label in valid_labels:
                raw_color = ps.cluster_colors.get(label, "#000000")
                try:
                    rgb = mcolors.to_rgb(raw_color)
                except ValueError:
                    rgb = parse_rgb_string(raw_color)
                cluster_colors[label] = rgb
        else:
            cmap = plt.get_cmap(ps.color_palette)
            for i, label in enumerate(valid_labels):
                cluster_colors[label] = cmap(i / max(1, len(valid_labels) - 1))

        return valid_labels, cluster_colors

    def render_plot():
        nonlocal cluster_colors

        nav_img_data = ps.nav_img.data
        if nav_img_data.ndim == 2:
            nav_img_rgb = np.stack([nav_img_data]*3, axis=-1)
        elif nav_img_data.ndim == 3 and nav_img_data.shape[0] == 3:
            nav_img_rgb = np.moveaxis(nav_img_data, 0, -1)
        elif nav_img_data.shape[-1] == 3:
            nav_img_rgb = nav_img_data
        else:
            raise ValueError("Unsupported navigation image format.")

        nav_img_rgb = nav_img_rgb.astype(np.float32)
        nav_img_rgb /= nav_img_rgb.max()

        phase = ps.labels.reshape(ps.height, ps.width)

        # Cluster overlay map
        cluster_map_rgb = np.zeros_like(nav_img_rgb)
        for label in valid_labels:
            mask = phase == label
            color = cluster_colors[label]
            for i in range(3):
                cluster_map_rgb[..., i][mask] = color[i]

        overlay_rgb = (
            alpha_cluster_map * cluster_map_rgb + (1 - alpha_cluster_map) * nav_img_rgb
        )
        overlay_rgb = np.clip(overlay_rgb, 0, 1)

        out.clear_output()
        with out:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].imshow(nav_img_rgb)
            axs[0].set_title("Navigation Image")
            axs[0].axis("off")

            axs[1].imshow(overlay_rgb)
            axs[1].set_title("Cluster Overlay")
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()
            save_fig(fig)

    # Initialize
    valid_labels, cluster_colors = build_color_map()

    # UI Elements
    layout_format = Layout(width="18%", style={"description_width": "initial"})
    color_pickers = [
        widgets.ColorPicker(value=mcolors.to_hex(cluster_colors[label]), description=f"cluster_{label}", layout=layout_format)
        for label in valid_labels
    ]

    out = widgets.Output()
    render_plot()

    def change_color(_):
        nonlocal cluster_colors
        updated_colors = {}
        for label, picker in zip(valid_labels, color_pickers):
            try:
                rgb = mcolors.to_rgb(picker.value)
            except ValueError:
                rgb = parse_rgb_string(picker.value)
            updated_colors[label] = rgb

        ps.cluster_colors = {label: mcolors.to_hex(rgb) for label, rgb in updated_colors.items()}
        cluster_colors = updated_colors
        render_plot()

    button = widgets.Button(description="Set", layout=Layout(width="auto"))
    button.on_click(change_color)

    color_rows = [widgets.HBox(color_pickers[i:i+5]) for i in range(0, len(color_pickers), 5)]
    color_box = widgets.VBox([*color_rows, button], layout=Layout(flex="2 1 0%", width="auto"))
    out_box = widgets.Box([out], layout=Layout(flex="8 1 0%", width="auto"))

    display(widgets.VBox([out_box, color_box]))

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
            row_index = list(change.new)
            display(weights.loc[row_index])
            for cluster in row_index:
                num_cpnt = len(weights.columns)
                fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=96)
                axs.bar(
                    np.arange(num_cpnt),
                    weights.loc[cluster].values,
                    width=0.6,
                )
                axs.set_xticks(np.arange(num_cpnt))
                axs.set_ylabel("Weight of component")
                axs.set_xlabel("Component number")
                axs.set_title(f"Weights for {cluster}")  # This adds clarity post-merge
                plt.show()
                save_fig(fig)


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


import matplotlib.colors as mcolors  # make sure this is imported

def view_clusters_sum_spectra(ps: PixelSegmenter, normalisation=True, spectra_range=(0, 8)):
    # Get actual cluster IDs in current label map (excluding -1)
    current_labels = ps.labels
    all_cluster_ids = sorted(int(c) for c in np.unique(current_labels) if c != -1)

    # Filter out clusters with no assigned pixels
    non_empty_clusters = [c for c in all_cluster_ids if np.sum(current_labels == c) > 0]
    cluster_options = [f"cluster_{c}" for c in non_empty_clusters]

    # UI widgets
    multi_select = widgets.SelectMultiple(options=cluster_options)
    plots_output = widgets.Output()
    profile_output = widgets.Output()
    figs = []

    # Helper to get the color for a cluster
    def get_cluster_color(cluster_id):
        default_color = "#000000"
        raw_color = ps.cluster_colors.get(cluster_id, default_color)
        try:
            return mcolors.to_rgb(raw_color)  # normalize to matplotlib RGB tuple
        except ValueError:
            # Fallback to parsing custom format
            return parse_rgb_string(raw_color)

    # Initial display: if no selection, show all clusters
    with plots_output:
        for cluster_id in non_empty_clusters:
            try:
                fig = ps.plot_binary_map_spectra_profile(
                    cluster_num=cluster_id,
                    normalisation=normalisation,
                    spectra_range=spectra_range,
                    color=get_cluster_color(cluster_id)  # 🎨 use color
                )
                figs.append(fig)
            except ValueError:
                print(f"Skipping empty cluster {cluster_id}")

    with profile_output:
        for cluster_id in non_empty_clusters:
            try:
                _, _, spectra_profile = ps.get_binary_map_spectra_profile(
                    cluster_num=cluster_id,
                    use_label=True
                )
                visual.plot_profile(
                    spectra_profile["energy"],
                    spectra_profile["intensity"],
                    ps.peak_list,
                    color=get_cluster_color(cluster_id)  # 🎨 use color here too
                )
            except ValueError:
                print(f"No spectra to plot for cluster {cluster_id}")

    # Callback for dropdown change
    def eventhandler(change):
        selected = change.new
        selected_cluster_ids = [int(name.split("_")[1]) for name in selected]

        plots_output.clear_output()
        profile_output.clear_output()

        with plots_output:
            if not selected_cluster_ids:
                for cluster_id in non_empty_clusters:
                    try:
                        fig = ps.plot_binary_map_spectra_profile(
                            cluster_num=cluster_id,
                            normalisation=normalisation,
                            spectra_range=spectra_range,
                            color=get_cluster_color(cluster_id)
                        )
                        figs.append(fig)
                    except ValueError:
                        print(f"Skipping empty cluster {cluster_id}")
            else:
                for cluster_id in selected_cluster_ids:
                    try:
                        fig = ps.plot_binary_map_spectra_profile(
                            cluster_num=cluster_id,
                            normalisation=normalisation,
                            spectra_range=spectra_range,
                            color=get_cluster_color(cluster_id)
                        )
                        figs.append(fig)
                    except ValueError:
                        print(f"Skipping empty cluster {cluster_id}")

        with profile_output:
            if not selected_cluster_ids:
                for cluster_id in non_empty_clusters:
                    try:
                        _, _, spectra_profile = ps.get_binary_map_spectra_profile(
                            cluster_num=cluster_id, use_label=True
                        )
                        visual.plot_profile(
                            spectra_profile["energy"],
                            spectra_profile["intensity"],
                            ps.peak_list,
                            color=get_cluster_color(cluster_id)
                        )
                    except ValueError:
                        print(f"No spectra to plot for cluster {cluster_id}")
            else:
                for cluster_id in selected_cluster_ids:
                    try:
                        _, _, spectra_profile = ps.get_binary_map_spectra_profile(
                            cluster_num=cluster_id, use_label=True
                        )
                        visual.plot_profile(
                            spectra_profile["energy"],
                            spectra_profile["intensity"],
                            ps.peak_list,
                            color=get_cluster_color(cluster_id)
                        )
                    except ValueError:
                        print(f"No spectra to plot for cluster {cluster_id}")

    multi_select.observe(eventhandler, names="value")

    # Display widgets
    display(multi_select)
    save_fig(figs)

    tab = widgets.Tab([plots_output, profile_output])
    tab.set_title(0, "clusters + spectra")
    tab.set_title(1, "spectra")
    display(tab)

#utility function
def get_non_empty_clusters(ps: PixelSegmenter):
    """Return a sorted list of cluster IDs that contain at least one pixel (excluding -1)."""
    return sorted(
        c for c in set(ps.labels.flatten())
        if c != -1 and np.sum(ps.labels == c) > 0
    )






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
                df_stats = ps.phase_stats(
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


def show_abundance_map(ps: PixelSegmenter, weights: pd.DataFrame, components: pd.DataFrame):
    def plot_rgb(ps, phases: List):
        shape = ps.get_binary_map_spectra_profile(0)[0].shape
        img = np.zeros((shape[0], shape[1], 3))

        for i, phase in enumerate(phases):
            if phase != 'None':
                if phase not in weights.columns:
                    continue

                cpnt_weights = weights[phase] / weights[phase].max()
                tmp = np.zeros(shape)

                valid_clusters = cpnt_weights.index
                for cluster_label in valid_clusters:
                    try:
                        cluster_id = int(''.join(filter(str.isdigit, cluster_label)))
                        if cluster_id not in ps.labels:
                            continue

                        idx = ps.get_binary_map_spectra_profile(cluster_id, use_label=True)[1]
                        tmp[idx] = cpnt_weights.loc[cluster_label]
                    except:
                        pass

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
        fig = plot_rgb(ps, phases=['None', 'None', 'None'])
        save_fig(fig)

    def dropdown_r_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(ps, phases=[change.new, dropdown_g.value, dropdown_b.value])
            save_fig(fig)

    def dropdown_g_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(ps, phases=[dropdown_r.value, change.new, dropdown_b.value])
            save_fig(fig)

    def dropdown_b_eventhandler(change):
        plots_output.clear_output()
        with plots_output:
            fig = plot_rgb(ps, phases=[dropdown_r.value, dropdown_g.value, change.new])
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
from ipywidgets import Button, Output, ToggleButtons, Dropdown, HBox, VBox, Layout, ColorPicker, HTML, Box
from IPython.display import clear_output

from IPython.display import HTML as IPyHTML


from collections import defaultdict


import pandas as pd
import random
import matplotlib.cm as cm
from ipywidgets import (
    Button, Layout, ToggleButtons, VBox, HBox, HTML, ColorPicker, Output
)
import plotly.graph_objs as go



#helper function for recomputing average compositions after creating new clusters



def compute_mu(ps):
    # pull out the elemental data
    X = ps.dataset.normalised_elemental_data
    # if it's an image (H, W, F), flatten to (H*W, F)
    if X.ndim == 3:
        H, W, F = X.shape
        X = X.reshape(H*W, F)
    labels = ps.labels.flatten()
    if X.shape[0] != labels.shape[0]:
        raise RuntimeError(f"Shape mismatch: X has {X.shape[0]} rows, labels has {labels.shape[0]}")
    # now build a dict of mean‐vectors
    unique_clusters = sorted(set(labels))
    mu = {}
    for k in unique_clusters:
        mask = labels == k
        if mask.any():
            mu[k] = X[mask].mean(axis=0)
        else:
            mu[k] = np.zeros(X.shape[1], dtype=X.dtype)
    ps.mu = mu


def refresh_cluster_statistics(ps):
    compute_mu(ps)

    # If using H x W images, reshape labels if needed
    if ps.labels.ndim == 1 and hasattr(ps, "height") and hasattr(ps, "width"):
        ps.labels = ps.labels.reshape((ps.height, ps.width))

    # Optionally invalidate or recompute prob_map
    if hasattr(ps, "prob_map"):
        ps.prob_map = None  # or recompute based on updated clustering

    # Force recompute of anything cached in get_binary_map_spectra_profile
    if hasattr(ps, "spectra_cache"):
        ps.spectra_cache = {}  # clear any old cache



def interactive_latent_plot(ps, ratio_to_be_shown=0.5, n_colours=30):
    
    """
    Interactive GUI for clustering in the latent space of a PixelSegmenter object.

    Allows:
    - Cluster selection by clicking or using lasso/rectangle
    - Recoloring selected clusters
    - Merging selected clusters (in-place)
    - Creating new clusters by selecting individual points

    Parameters:
    -----------
    ps : PixelSegmenter
        The object to be visualized.
    ratio_to_be_shown : float
        Fraction of latent points to display (for performance).
    n_colours : int
        Number of color swatches in the palette.
    """
    



    
    original_palette = '%s' % ps.color_palette
    original_labels = ps.labels.copy()
    

    def update_cluster_colors():
        valid_labels = sorted(set(ps.labels.flatten()) - {-1})
        cmap = cm.get_cmap(ps.color_palette)

        # Only assign new colors to clusters that do not already have a manually assigned one
        for i, label in enumerate(valid_labels):
            if label not in ps.manual_cluster_colors:
                color = "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in cmap(i / max(1, len(valid_labels) - 1))[:3]))
                ps.cluster_colors[label] = color
            else:
                ps.cluster_colors[label] = ps.manual_cluster_colors[label]

    update_cluster_colors()

    selected_point_indices = set()
    new_cluster_mode = [False]

    original_cluster_colors = ps.cluster_colors.copy()

    def get_color(cid):
        if cid == -1:
            return '#999999'
        return (
            ps.manual_cluster_colors.get(cid)
            or ps.cluster_colors.get(cid)
            or original_cluster_colors.get(cid, '#cccccc')
        )

    cmap = cm.get_cmap(ps.color_palette)
    norm = ps.color_norm
    latent = ps.latent
    labels = ps.labels
    sampled_indices = random.sample(range(latent.shape[0]), int(latent.shape[0] * ratio_to_be_shown)) if ratio_to_be_shown != 1.0 else list(range(latent.shape[0]))
    df = pd.DataFrame(latent[sampled_indices], columns=["x", "y"])
    df["cluster"] = labels.flatten()[sampled_indices]
    df["index"] = sampled_indices

    selected_clusters = set()
    out = Output()
    confirm_out = Output()

    fig = go.FigureWidget()
    fig.update_layout(
        title="Interactive Latent Space",
        xaxis_title="Latent X",
        yaxis_title="Latent Y",
        plot_bgcolor='white',
        height=600,
        width=600,
        dragmode="select"
    )
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', zerolinecolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', zerolinecolor='lightgrey')

    def plot():
        nonlocal df
        current_labels = ps.labels.flatten()
        df = pd.DataFrame(latent[sampled_indices], columns=["x", "y"])
        df["cluster"] = current_labels[sampled_indices]
        df["index"] = sampled_indices

        with fig.batch_update():
            fig.data = []

            colors = df["cluster"].map(lambda cid: ps.manual_cluster_colors.get(cid, get_color(cid)))
            fig.add_trace(go.Scattergl(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(size=6, color=colors),
                name="Clusters",
                customdata=np.stack([df["cluster"], df["index"]], axis=1),
                hovertemplate="Cluster: %{customdata[0]}<br>Index: %{customdata[1]}<extra></extra>"
            ))

            trace = fig.data[0]
            trace.on_click(on_point_click)
            trace.on_selection(on_select)

    def on_point_click(trace, points, selector):
        for i in points.point_inds:
            cluster_id = int(trace.customdata[i][0])
            if cluster_id == -1:
                continue
            selected_clusters.add(cluster_id)

            color_hex = get_color(cluster_id)
            r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            rgb_str = f'rgb({r}, {g}, {b})'

            color_info_box.value = (
                f"<div style='padding:4px;'>"
                f"<b>Cluster {cluster_id}</b><br>"
                f"<div style='width:50px; height:20px; background:{color_hex}; border:1px solid #000;'></div><br>"
                f"<b>HEX:</b> {color_hex}<br><b>RGB:</b> {rgb_str}"
                f"</div>"
            )
        with out:
            print(f"Clicked cluster(s): {sorted(selected_clusters)}")

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
                out.clear_output()
                print("⚠️ No points selected.")
            return

        nonlocal labels
        current_labels = ps.labels.flatten()
        existing_clusters = set(current_labels)
        new_cluster_id = max(existing_clusters) + 1

        count = 0
        for idx in selected_point_indices:
            if include_noise_toggle.value == "Include -1" or current_labels[idx] != -1:
                current_labels[idx] = new_cluster_id
                count += 1

        if count == 0:
            with out:
                out.clear_output()
                print("⚠️ No eligible points updated.")
            return

        ps.labels = current_labels.reshape(ps.height * ps.width, 1)
        labels = ps.labels
        ps.n_components = len(set(labels.flatten()))
        selected_point_indices.clear()
        new_cluster_mode[0] = False
        update_cluster_colors()
        refresh_cluster_statistics(ps)
        with out:
            out.clear_output()
            print(f"✅ Created new cluster {new_cluster_id} with {count} points.")
        plot()

    def on_merge_clicked(b):
        if not selected_clusters:
            with out:
                out.clear_output()
                print("⚠️ No clusters selected to merge.")
            return

        selected_clusters.difference_update([-1])
        if len(selected_clusters) <= 1:
            with out:
                out.clear_output()
                print("⚠️ Need at least two clusters to merge.")
            return

        target_id = min(selected_clusters)
        label_array = ps.labels.flatten()
        merge_count = 0
        for cluster_id in selected_clusters:
            if cluster_id != target_id:
                merge_count += np.sum(label_array == cluster_id)
                label_array[label_array == cluster_id] = target_id

        ps.labels = label_array.reshape(ps.height * ps.width, 1)
        
        used_labels = set(label_array)
        ps.cluster_colors = {k: v for k, v in ps.cluster_colors.items() if k in used_labels}
        update_cluster_colors()
        
        
        ps.n_components = len(set(ps.labels.flatten()))
        selected_clusters.clear()
        update_cluster_colors()
        refresh_cluster_statistics(ps)

        with out:
            out.clear_output()
            print(f"✅ Merged clusters into cluster {target_id}.")
            print(f"   Total points reassigned: {merge_count}")
        plot()

    def on_reset_clicked(b):
        confirm_out.clear_output()
        with confirm_out:
            print("⚠️ Are you sure you want to reset everything?")
            confirm = Button(description="Yes, Reset", button_style='danger')
            cancel = Button(description="Cancel", button_style='info')

            def do_reset(btn):
                nonlocal labels
                selected_clusters.clear()
                ps.manual_cluster_colors.clear()
                selected_point_indices.clear()
                new_cluster_mode[0] = False
                ps.color_palette = original_palette
                ps.labels = original_labels.copy()
                labels = ps.labels
                ps.n_components = len(set(labels.flatten()))
                update_cluster_colors()
                refresh_cluster_statistics(ps)
                confirm_out.clear_output()
                out.clear_output()
                print("✅ Full reset complete.")
                plot()

            def cancel_reset(btn):
                confirm_out.clear_output()
                with out:
                    out.clear_output()
                    print("❎ Reset cancelled.")

            confirm.on_click(do_reset)
            cancel.on_click(cancel_reset)
            display(HBox([confirm, cancel]))

    def on_recolor_clicked(b):
        if not selected_clusters or not selected_color[0]:
            with out:
                print("Please select clusters and a color first.")
            return

        for cluster_id in selected_clusters:
            if cluster_id == -1:
                continue
            ps.manual_cluster_colors[cluster_id] = selected_color[0]  # persist manual color
            ps.cluster_colors[cluster_id] = selected_color[0]

        selected_clusters.clear()
        with out:
            out.clear_output()
            print("✅ Recolored selected clusters.")
        plot()

    def on_clear_selection_clicked(b):
        selected_clusters.clear()
        with out:
            out.clear_output()
            print("🧼 Cleared selected clusters.")

    standard_button_layout = Layout(width='150px', height='35px')
    recolor_button = Button(description="Recolour Selection", layout=standard_button_layout)
    merge_button = Button(description="Merge Clusters", layout=standard_button_layout)
    reset_button = Button(description="Reset", layout=standard_button_layout)
    clear_selection_button = Button(description="Clear Selection", layout=standard_button_layout)
    select_points_button = Button(description="Select Points", layout=standard_button_layout)
    create_cluster_button = Button(description="Create Cluster", layout=standard_button_layout)
    include_noise_toggle = ToggleButtons(options=["Include -1", "Exclude -1"], description="Include/Exclude Noise:", value="Exclude -1")
    color_info_box = HTML(value="<b>No cluster selected</b>")
    color_picker = ColorPicker(description='Custom Color:', value='#00ff00')
    selected_color = [None]

    color_picker.observe(lambda c: selected_color.__setitem__(0, c['new']), names='value')

    color_buttons = []
    for rgba in [cmap(i / (n_colours - 1)) for i in range(n_colours)]:
        r, g, b_, _ = rgba
        rgb_str = f'rgb({int(r*255)}, {int(g*255)}, {int(b_*255)})'
        btn = Button(layout=Layout(width='30px', height='30px'), tooltip=rgb_str, style={'button_color': rgb_str})
        btn.on_click(lambda b, rgb=rgb_str: selected_color.__setitem__(0, rgb))
        color_buttons.append(btn)

    color_selector_ui = VBox([HBox(color_buttons[:15]), HBox(color_buttons[15:])])

    merge_button.on_click(on_merge_clicked)
    reset_button.on_click(on_reset_clicked)
    recolor_button.on_click(on_recolor_clicked)
    select_points_button.on_click(on_select_points_clicked)
    create_cluster_button.on_click(on_create_cluster_clicked)
    clear_selection_button.on_click(on_clear_selection_clicked)

    plot()

    display(VBox([
        HBox([recolor_button, merge_button,select_points_button, create_cluster_button]),
        HBox([include_noise_toggle]),
        color_info_box,
        color_selector_ui,
        color_picker,
        fig,
        HBox([clear_selection_button, reset_button]),
        confirm_out,
        out
    ]))


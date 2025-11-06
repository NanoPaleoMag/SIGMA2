#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sigma.utils.load import SEMDataset, IMAGEDataset, PIXLDataset, AZTECDataset
from sigma.utils.loadtem import TEMDataset
from sigma.utils.visualisation import make_colormap
from sigma.src.utils import k_factors_120kV

from typing import List, Dict, Union
import hyperspy.api as hs
import numpy as np
import pandas as pd
# import hdbscan
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, Birch, HDBSCAN
from skimage import measure
from scipy import fftpack
from skimage.transform import resize

from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import pickle
import os

from tifffile import imsave

from sklearn.utils.extmath import randomized_svd




class PixelSegmenter(object):
    def __init__(
        self,
        latent: np.ndarray,
        dataset: Union[SEMDataset, TEMDataset, IMAGEDataset],
        method: str = "BayesianGaussianMixture",
        method_args: Dict = {"n_components": 8, "random_state": 4},
        
    ):
        
        self.cluster_colors = {}  # dict of cluster_id -> RGB string like 'rgb(255,0,0)'
        self.latent = latent
        self.dataset = dataset
        self.dataset_norm = dataset.normalised_elemental_data
        self.method = method
        self.method_args = method_args
        self.height = self.dataset_norm.shape[0]
        self.width = self.dataset_norm.shape[1]
        self.manual_cluster_colors={}

        # Set spectra and nav_img signal to the corresponding ones
        if type(dataset) not in [IMAGEDataset, PIXLDataset]:
            if self.dataset.spectra_bin is not None:
                self.spectra = self.dataset.spectra_bin
            else:
                self.spectra = self.dataset.spectra

            if self.dataset.nav_img_bin is not None:
                self.nav_img = self.dataset.nav_img_bin
            else:
                self.nav_img = self.dataset.nav_img
            if type(dataset)!=AZTECDataset:
                ### Get energy_axis ###
                size = self.spectra.axes_manager[2].size
                scale = self.spectra.axes_manager[2].scale
                offset = self.spectra.axes_manager[2].offset
                self.energy_axis = [((a * scale) + offset) for a in range(0, size)]

        ### Train the model ###
        if self.method == "GaussianMixture":
            self.model = GaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "BayesianGaussianMixture":
            self.model = BayesianGaussianMixture(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_components"]
        elif self.method == "Kmeans":
            self.model = KMeans(**method_args).fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "Birch":
            self.model = Birch(**method_args).partial_fit(self.latent)
            self.n_components = self.method_args["n_clusters"]
        elif self.method == "HDBSCAN":
            self.model = HDBSCAN(**method_args)
            self.labels = self.model.fit_predict(self.latent)
            self.n_components = int(self.labels.max()) + 1 
            
        if self.method != "HDBSCAN":
            self.labels = self.model.predict(self.latent)
            
        ### calculate cluster probability maps ###
        means = []
        dataset_ravel = self.dataset_norm.reshape(-1, self.dataset_norm.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(self.labels == i)[0]].mean(axis=0)
            means.append(mean.reshape(1, -1))
        mu = np.concatenate(means, axis=0)

        if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
            self.prob_map = self.model.predict_proba(self.latent)
        elif self.method == "HDBSCAN":
            self.prob_map = self.model.probabilities_

        self.mu = mu

        ### Calcuate peak_dict ###
        self.peak_dict = dict()
        for element in hs.material.elements:
            if element[0] == "Li":
                continue
            for character in element[1].Atomic_properties.Xray_lines:
                peak_name = element[0]
                char_name = character[0]
                key = f"{peak_name}_{char_name}"
                self.peak_dict[key] = character[1].energy_keV

        self.peak_list = self.dataset.feature_list

        # Set color for phase visualisation
        if self.n_components <= 10:
            self._color_palette = "tab10"
            self.color_palette = "tab10"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=9)
        else:
            self._color_palette = "nipy_spectral"
            self.color_palette = "nipy_spectral"
            self.color_norm = mpl.colors.Normalize(vmin=0, vmax=self.n_components - 1)

    def set_color_palette(self, cmap):
        self.color_palette = cmap

    def set_feature_list(self, new_list):
        self.peak_list = new_list
        self.dataset.set_feature_list(new_list)

    @staticmethod
    def bic(
        latent,
        n_components=20,
        model="BayesianGaussianMixture",
        model_args={"random_state": 6},
    ):
        def _n_parameters(model):
            """Return the number of free parameters in the model."""
            _, n_features = model.means_.shape
            if model.covariance_type == "full":
                cov_params = model.n_components * n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "diag":
                cov_params = model.n_components * n_features
            elif model.covariance_type == "tied":
                cov_params = n_features * (n_features + 1) / 2.0
            elif model.covariance_type == "spherical":
                cov_params = model.n_components
            mean_params = n_features * model.n_components
            return int(cov_params + mean_params + model.n_components - 1)

        bic_list = []
        for i in range(n_components):
            if model == "BayesianGaussianMixture":
                GMM = BayesianGaussianMixture(n_components=i + 1, **model_args).fit(
                    latent
                )
            elif model == "GaussianMixture":
                GMM = GaussianMixture(n_components=i + 1, **model_args).fit(latent)
            bic = -2 * GMM.score(latent) * latent.shape[0] + _n_parameters(
                GMM
            ) * np.log(latent.shape[0])
            bic_list.append(bic)
        return bic_list

    #################
    # Data Analysis #--------------------------------------------------------------
    #################


    def get_binary_map_spectra_profile(
        self,
        cluster_num=1,
        use_label=False,
        threshold=0.8,
        denoise=False,
        keep_fraction=0.13,
        binary_filter_threshold=0.2,
    ):
        # Determine spatial shape from spectra
        spectra_shape = tuple(self.spectra.data.shape[:2])
        n_pixels = int(np.prod(spectra_shape))

        # --- Step 1: Get binary mask from soft or hard clustering ---
        binary_map_flat = None
        # Helper for final validation & reshape
        def validate_and_return(flat_map):
            flat_map = np.asarray(flat_map)
            if flat_map.size != n_pixels:
                raise ValueError(
                    f"Mismatch: label size ({flat_map.size}) doesn't match spectra shape {spectra_shape} (n_pixels={n_pixels})"
                )
            return flat_map.astype(int)

        # Try soft mask from prob_map (if present)
        if not use_label:
            use_soft_mask = False
            if hasattr(self, "prob_map") and self.prob_map is not None:
                pm = np.asarray(self.prob_map)
                # If prob_map is 1D but length equals n_pixels, treat as (n_pixels, 1)
                if pm.ndim == 1:
                    if pm.size == n_pixels:
                        pm = pm.reshape(n_pixels, 1)
                    else:
                        # prob_map seems incompatible with spectra length
                        pm = None
                if pm is not None and pm.ndim == 2:
                    # safe check for cluster_num bound
                    if 0 <= int(cluster_num) < pm.shape[1]:
                        phase = pm[:, int(cluster_num)]
                        use_soft_mask = True

            # fallback: model.predict_proba if available
            if not use_soft_mask and hasattr(self, "model") and hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(self.latent)
                    proba = np.asarray(proba)
                    if proba.ndim == 1 and proba.size == n_pixels:
                        proba = proba.reshape(n_pixels, 1)
                    if proba.ndim == 2 and 0 <= int(cluster_num) < proba.shape[1]:
                        phase = proba[:, int(cluster_num)]
                        use_soft_mask = True
                except (AttributeError, IndexError, ValueError):
                    use_soft_mask = False

            if use_soft_mask:
                # ensure phase length matches n_pixels
                phase = np.asarray(phase)
                if phase.size != n_pixels:
                    raise ValueError(f"probability vector length {phase.size} != expected n_pixels {n_pixels}")
                binary_map_flat = (phase > threshold).astype(int)
            else:
                # Try to use hard labels
                if hasattr(self, "labels"):
                    lbls = np.asarray(self.labels)
                    # labels could already be flat of length n_pixels
                    if lbls.size == n_pixels:
                        binary_map_flat = (lbls == cluster_num).astype(int)
                    # or labels could be 2D with same spatial shape
                    elif lbls.shape == spectra_shape:
                        binary_map_flat = (lbls.ravel() == cluster_num).astype(int)
                    else:
                        # If labels are not pixel-level, we can't use them directly
                        raise ValueError(
                            f"self.labels found but shape {lbls.shape} incompatible with spectra_shape {spectra_shape}."
                        )
                else:
                    raise ValueError(f"Could not find soft or hard cluster mask for cluster {cluster_num}")
        else:
            # use_label=True: compare labels to cluster_num (allow string/int labels)
            if not hasattr(self, "labels"):
                raise ValueError("use_label=True but self.labels not present")
            lbls = np.asarray(self.labels)
            if lbls.size == n_pixels:
                binary_map_flat = (lbls == cluster_num).astype(int)
            elif lbls.shape == spectra_shape:
                binary_map_flat = (lbls.ravel() == cluster_num).astype(int)
            else:
                raise ValueError(f"self.labels shape {lbls.shape} incompatible with spectra_shape {spectra_shape}.")

        # Validate & reshape to 2D
        binary_map_flat = validate_and_return(binary_map_flat)
        binary_map = binary_map_flat.reshape(spectra_shape)
        binary_map_indices = np.where(binary_map == 1)

        # --- Step 2: Gather feature values at the masked pixel locations ---
        if binary_map_indices[0].size == 0:
            raise ValueError(f"No pixels found in cluster {cluster_num}.")

        x_y_indices = tuple(zip(*binary_map_indices))

        # --- Step 3: Compute intensity profile from data ---
        if isinstance(self.dataset, (IMAGEDataset, PIXLDataset)):
            maps = self.dataset.chemical_maps_bin if self.dataset.chemical_maps_bin is not None else self.dataset.chemical_maps
            assert maps.shape[-1] == len(self.dataset.feature_list), \
                f"Shape mismatch: maps.shape[-1]={maps.shape[-1]} vs feature_list={len(self.dataset.feature_list)}"

            total_spectra_profiles = np.array([maps[x, y, :] for x, y in x_y_indices])
            energy_axis = self.dataset.feature_list
        else:
            total_spectra_profiles = np.array([self.spectra.data[x, y, :] for x, y in x_y_indices])
            size = self.spectra.axes_manager[2].size
            scale = self.spectra.axes_manager[2].scale
            offset = self.spectra.axes_manager[2].offset
            energy_axis = [((a * scale) + offset) for a in range(0, size)]

        # --- Step 4: Compute mean intensity ---
        element_intensity_mean = total_spectra_profiles.mean(axis=0)

        spectra_profile = pd.DataFrame(
            data=np.column_stack([energy_axis, element_intensity_mean]),
            columns=["energy", "intensity"]
        )

        return binary_map, binary_map_indices, spectra_profile

    def get_all_spectra_profile(self, normalised=True):
        spectra_profiles = []

        # Use actual cluster labels that exist in self.labels
        unique_clusters = np.unique(self.labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # skip outliers/noise

        for cluster_id in unique_clusters:
            try:
                _, _, spectra_profile = self.get_binary_map_spectra_profile(
                    cluster_num=cluster_id, use_label=True
                )
                spectra_profiles.append(spectra_profile["intensity"])
            except Exception as e:
                print(f"Skipping cluster {cluster_id} due to error: {e}")

        if not spectra_profiles:
            raise ValueError("No valid spectra profiles found for any cluster.")

        spectra_profiles = np.vstack(spectra_profiles)

        if normalised:
            spectra_profiles *= 1 / spectra_profiles.max(axis=1, keepdims=True)

        return spectra_profiles, unique_clusters

    def get_unmixed_spectra_profile(
        self,
        clusters_to_be_calculated="All",
        n_components="All",
        normalised=True,
        method="NMF",
        method_args={},
    ):
        

        assert method == "NMF", "Only NMF is supported currently."

        # Get all spectra profiles (shape: clusters x features)
        spectra_profiles, cluster_ids = self.get_all_spectra_profile(normalised)
        
        
        # Wrap into DataFrame with numeric cluster index
        spectra_profiles_ = pd.DataFrame(
            spectra_profiles.T,  # shape: features x clusters
            columns=cluster_ids,
        )

        # Subset clusters to calculate
        if clusters_to_be_calculated != "All":
            spectra_profiles_ = spectra_profiles_[clusters_to_be_calculated]

        # Determine number of NMF components
        if n_components == "All":
            n_components = spectra_profiles_.shape[1]

        # Fit NMF
        model = NMF(n_components=n_components, **method_args)
        weights = model.fit_transform(spectra_profiles_.to_numpy().T)  # shape: clusters x components
        components = model.components_  # shape: components x features

        # Build DataFrames with clean, robust indexing
        cluster_labels = [f"cluster_{i}" for i in spectra_profiles_.columns]
        weights_df = pd.DataFrame(
            weights.round(3),
            columns=[f"w_{i}" for i in range(n_components)],
            index=cluster_labels,
        )
        components_df = pd.DataFrame(
            components.T.round(3),
            columns=[f"cpnt_{i}" for i in range(n_components)],
        )

        self.NMF_recon_error = model.reconstruction_err_

        return weights_df, components_df


        if clusters_to_be_calculated != "All":
            num_inputs = len(clusters_to_be_calculated)
        else:
            num_inputs = self.n_components

        if n_components == "All":
            n_components = num_inputs

        assert method == "NMF"
        if method == "NMF":
            model = NMF(n_components=n_components, **method_args)

        spectra_profiles = self.get_all_spectra_profile(normalised)
        spectra_profiles_ = pd.DataFrame(
            spectra_profiles.T, columns=range(spectra_profiles.shape[0])
        )

        if clusters_to_be_calculated != "All":
            spectra_profiles_ = spectra_profiles_[clusters_to_be_calculated]

        weights = model.fit_transform(spectra_profiles_.to_numpy().T)
        components = model.components_
        self.NMF_recon_error = model.reconstruction_err_

        weights = pd.DataFrame(
            weights.round(3),
            columns=[f"w_{component_num}" for component_num in range(n_components)],
            index=[f"cluster_{cluster_num}" for cluster_num in spectra_profiles_],
        )
        components = pd.DataFrame(
            components.T.round(3),
            columns=[f"cpnt_{component_num}" for component_num in range(n_components)],
        )

        return weights, components

    def get_masked_spectra(
        self,
        cluster_num,
        threshold=0.8,
        denoise=False,
        keep_fraction=0.13,
        binary_filter_threshold=0.2,
        **binary_filter_args,
    ):

        phase = self.model.predict_proba(self.latent)[:, cluster_num]

        if denoise == False:
            binary_map_indices = np.where(
                phase.reshape(self.height, self.width) <= threshold
            )

        else:
            filtered_img = np.where(phase < threshold, 0, 1).reshape(
                self.height, self.width
            )
            image_fft = fftpack.fft2(filtered_img)
            image_fft2 = image_fft.copy()

            # Set r and c to be the number of rows and columns of the array.
            r, c = image_fft2.shape

            # Set to zero all rows with indices between r*keep_fraction and
            # r*(1-keep_fraction):
            image_fft2[int(r * keep_fraction) : int(r * (1 - keep_fraction))] = 0

            # Similarly with the columns:
            image_fft2[:, int(c * keep_fraction) : int(c * (1 - keep_fraction))] = 0

            # Transformed the filtered image back to real space
            image_new = fftpack.ifft2(image_fft2).real

            binary_map_indices = np.where(image_new > binary_filter_threshold)

        # Get spectra profile in the filtered phase region
        x_id = binary_map_indices[0].reshape(-1, 1)
        y_id = binary_map_indices[1].reshape(-1, 1)
        x_y = np.concatenate([x_id, y_id], axis=1)
        x_y_indices = tuple(map(tuple, x_y))

        shape = self.spectra.inav[0, 0].data.shape
        masked_spectra = self.spectra.deepcopy()
        for x_y_index in x_y_indices:
            masked_spectra.data[x_y_index] = np.zeros(shape)

        return masked_spectra

    def phase_stats(
        self, cluster_num, element_peaks=["Fe_Ka", "O_Ka"], binary_filter_args={}
    ):
        """

        Parameters
        ----------
        binary_map : np.array
            The filtered binary map for analysis.
        element_peaks : dict(), optional
            Determine whether the output includes the elemental intensity from 
            the origianl spectra signal. The default is ['Fe_Ka','O_Ka'].
        binary_filter_args : dict()
            Determine the parameters to generate the binary for the analysis.

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe whcih contains all statistical inforamtion of phase distribution.
            These include 'area','equivalent_diameter', 'major_axis_length','minor_axis_length','min_intensity','mean_intensity','max_intensity'

        """

        if binary_filter_args == {}:
            use_label = True
        else:
            use_label = False
        binary_map, _, _ = self.get_binary_map_spectra_profile(
            cluster_num, use_label=use_label, **binary_filter_args
        )
        pixel_to_um = self.spectra.axes_manager[0].scale
        prop_list = [
            "area",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "min_intensity",
            "mean_intensity",
            "max_intensity",
        ]

        label_binary_map = measure.label(binary_map, connectivity=2)
        element_maps = self.dataset.get_feature_maps()

        # Create a dataframe to record all statical information
        stat_info = dict()

        # for each element, create an individual element intensity statics
        for i, element in enumerate(element_peaks):
            element_idx = self.dataset.feature_dict[element]
            clusters = measure.regionprops(
                label_image=label_binary_map,
                intensity_image=element_maps[:, :, element_idx],
            )

            # for the first iteration, record everything apart from elemental intensity i.e. area, length ...
            if i == 0:
                for prop in prop_list:
                    if prop == "area":
                        stat_info[f"{prop} (um^2)"] = [
                            cluster[prop] * pixel_to_um ** 2 for cluster in clusters
                        ]

                    elif prop in [
                        "equivalent_diameter",
                        "major_axis_length",
                        "minor_axis_length",
                    ]:
                        stat_info[f"{prop} (um)"] = [
                            cluster[prop] * pixel_to_um for cluster in clusters
                        ]

                    elif prop in ["min_intensity", "mean_intensity", "max_intensity"]:
                        stat_info[f"{prop}_{element}"] = [
                            cluster[prop] for cluster in clusters
                        ]

            # for the remaining iteration, only add elemental intensity into the dict()
            else:
                for prop in ["min_intensity", "mean_intensity", "max_intensity"]:
                    stat_info[f"{prop}_{element}"] = [
                        cluster[prop] for cluster in clusters
                    ]

        return pd.DataFrame(data=stat_info).round(3)
    
    def cluster_quantification(self,
                               cluster_num:int,
                               elements:List,
                               k_factors:List[float]=None,
                               composition_units:str='atomic',
                               use_label:bool=True)-> pd.DataFrame:
        
        # get indices of the specified cluster
        binary_map, binary_map_indices, _ = self.get_binary_map_spectra_profile(cluster_num=cluster_num,use_label=use_label)
        indices = np.column_stack(binary_map_indices)
        indices = tuple(map(tuple, indices))
        
        # set elements for quantification
        spectra_raw = self.dataset.spectra_raw
        spectra_raw.metadata.Sample.xray_lines = elements
        intensities = spectra_raw.get_lines_intensity()
        
        if k_factors is None:
            try:
                k_factors = [k_factors_120kV[el] for el in elements]
            except KeyError:
                print('The k factor is not in the database.')
        
        compositions = spectra_raw.quantification(intensities, method='CL',factors=k_factors,composition_units='atomic')
        cluster_element_intensities = [c.data[binary_map.astype(bool)] for c in compositions]
        cluster_element_intensities = np.column_stack(cluster_element_intensities)
        
        return pd.DataFrame(cluster_element_intensities, columns = [el.split('_')[0] for el in elements])

        
        
    #################
    # Visualization #--------------------------------------------------------------
    #################

    def plot_latent_space(self, color=True, cmap=None):
        cmap = self.color_palette if cmap is None else cmap

        fig, axs = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
        label = self.labels

        if color:
            axs.scatter(
                self.latent[:, 0],
                self.latent[:, 1],
                c=label,
                s=2.0,
                zorder=2,
                alpha=0.15,
                linewidths=0,
                cmap=cmap,
                norm=self.color_norm,
            )

            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                i = 0
                for pos, covar, w in zip(
                    self.model.means_, self.model.covariances_, self.model.weights_
                ):
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.14,
                        facecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        edgecolor="None",
                        zorder=-10,
                    )
                    self.draw_ellipse(
                        pos,
                        covar,
                        alpha=0.0,
                        edgecolor=plt.cm.get_cmap(cmap)(
                            i * (self.n_components - 1) ** -1
                        ),
                        facecolor="None",
                        zorder=-9,
                        lw=0.25,
                    )
                    i += 1
        else:
            axs.scatter(
                self.latent[:, 0],
                self.latent[:, 1],
                c="k",
                s=1.0,
                zorder=2,
                alpha=0.15,
                linewidths=0,
            )

        for axis in ["top", "bottom", "left", "right"]:
            axs.spines[axis].set_linewidth(1.5)
        plt.show()
        return fig

    def draw_ellipse(self, position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 3):
            ax.add_patch(
                Ellipse(position, width=nsig*width, height=nsig*height, angle=angle, **kwargs)
            )

    def plot_cluster_distribution(self, save=None, **kwargs):
        labels = self.model.predict(self.latent)
        means = []
        dataset_ravel = self.dataset_norm.reshape(-1, self.dataset_norm.shape[2])
        for i in range(self.n_components):
            mean = dataset_ravel[np.where(labels == i)[0]].mean(axis=0)
            means.append(mean.reshape(1, -1))
        mu = np.concatenate(means, axis=0)

        fig, axs = plt.subplots(
            self.n_components,
            2,
            figsize=(14, self.n_components * 4.2),
            dpi=96,
            **kwargs,
        )
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        for i in range(self.n_components):
            if self.method in ["GaussianMixture", "BayesianGaussianMixture"]:
                prob_map_i = self.prob_map[:, i]
            else:
                prob_map_i = np.where(labels == i, 1, 0)
            im = axs[i, 0].imshow(
                prob_map_i.reshape(self.height, self.width), cmap="viridis"
            )
            axs[i, 0].set_title("Probability of each pixel for cluster " + str(i))

            axs[i, 0].axis("off")
            cbar = fig.colorbar(im, ax=axs[i, 0], shrink=0.9, pad=0.025)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=10, size=0)

            if self.n_components <= 10:
                axs[i, 1].bar(
                    self.dataset.feature_list,
                    mu[i],
                    width=0.6,
                    color=plt.cm.get_cmap(self.color_palette)(i * 0.1),
                )
            else:
                axs[i, 1].bar(
                    self.dataset.feature_list,
                    mu[i],
                    width=0.6,
                    color=plt.cm.get_cmap(self.color_palette)(
                        i * (self.n_components - 1) ** -1
                    ),
                )

            axs[i, 1].set_title("Mean value for cluster " + str(i))

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        plt.show()

        if save is not None:
            fig.savefig(save, bbox_inches="tight", pad_inches=0.01)

    def plot_single_cluster_distribution(self, cluster_num, spectra_range=(0, 8), color=None):
        ncols, figsize = 3, (13, 2.5)
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, dpi=120)
        fig.subplots_adjust(hspace=0.35, wspace=0.1)

        formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))

        # --- Prob map or binary label mask ---
        if hasattr(self, "prob_map") and self.prob_map is not None and cluster_num < self.prob_map.shape[1]:
            try:
                prob_map_i = np.asarray(self.prob_map[:, cluster_num], dtype=float)
                title_prefix = "Pixel-wise probability"
            except Exception as e:
                raise ValueError(f"Invalid prob_map for cluster {cluster_num}: {e}")
        elif hasattr(self, "labels"):
            prob_map_i = (self.labels == cluster_num).astype(float)
            if self.method == "HDBSCAN" and hasattr(self, "prob_map") and self.prob_map is not None:
                try:
                    prob_map_vals = np.asarray(self.prob_map[:, cluster_num], dtype=float)
                    prob_map_i *= prob_map_vals
                except Exception as e:
                    print(f"⚠️ Warning: Could not apply HDBSCAN prob_map to binary mask — {e}")
            title_prefix = "Binary label mask"
        else:
            raise ValueError(f"Cluster {cluster_num} could not be found in labels or prob_map")

        im = axs[0].imshow(prob_map_i.reshape(self.height, self.width), cmap="viridis")
        axs[0].set_title(f"{title_prefix} for cluster {cluster_num}")
        axs[0].axis("off")
        cbar = fig.colorbar(im, ax=axs[0], shrink=0.9, pad=0.025)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=10, size=0)

        # --- Mean feature bar plot ---
        mu_values = None
        if isinstance(getattr(self, "mu", None), dict):
            if cluster_num in self.mu:
                mu_values = self.mu[cluster_num]
        elif hasattr(self, "mu") and hasattr(self.mu, "__len__"):
            if 0 <= cluster_num < len(self.mu):
                mu_values = self.mu[cluster_num]

        if mu_values is not None:
            plot_color = (
                color if color is not None else
                plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1 if self.n_components <= 10 else cluster_num * (self.n_components - 1) ** -1)
            )

            axs[1].bar(self.dataset.feature_list, mu_values, width=0.6, color=plot_color)
            for i, feat in enumerate(self.dataset.feature_list):
                y = mu_values[i] + mu_values.max() * 0.03 if mu_values[i] > 0 else mu_values[i] - mu_values.max() * 0.08
                axs[1].text(i - len(feat) * 0.11, y, feat, fontsize=8)

            axs[1].set_xticks([])
            axs[1].set_xticklabels([])
            for spine in ["top", "right"]:
                axs[1].spines[spine].set_linewidth(0)
            if (mu_values < 0).any():
                axs[1].set_ylim(mu_values.min() * 1.2, mu_values.max() * 1.2)
                axs[1].spines["bottom"].set_position(("data", 0.0))
            else:
                axs[1].set_ylim(None, mu_values.max() * 1.2)
            axs[1].set_title(f"Mean feature value for cluster {cluster_num}")
        else:
            axs[1].text(0.5, 0.5, f"No `mu` for cluster {cluster_num}", ha="center")
            axs[1].set_axis_off()

        # --- Summed spectra ---
        try:
            if isinstance(self.dataset, (IMAGEDataset, PIXLDataset)):
                chemical_maps = self.dataset.chemical_maps if self.dataset.chemical_maps_bin is None else self.dataset.chemical_maps_bin
                avg_intensity = chemical_maps.mean(axis=(0, 1)).astype(np.float32)
                _, num_pixels, spectra_profile = self.get_binary_map_spectra_profile(cluster_num)
                num_pixels = len(num_pixels[0])
                mean_intensity = spectra_profile["intensity"].to_numpy(dtype=np.float32) / num_pixels

                axs[2].bar(self.dataset.feature_list, avg_intensity, width=0.7, facecolor="None",
                           edgecolor=sns.color_palette()[0], linestyle="dotted", linewidth=1,
                           zorder=10, label="Avg. raw spectrum")

                axs[2].bar(self.dataset.feature_list, mean_intensity, width=0.6, linewidth=1,
                           color=plot_color)

                for i in range(len(self.dataset.feature_list)):
                    y = mean_intensity[i] + mean_intensity.max() * 0.03
                    y_avg = avg_intensity[i] + mean_intensity.max() * 0.03
                    axs[2].text(i - len(self.dataset.feature_list[i]) * 0.11, max(y, y_avg), self.dataset.feature_list[i], fontsize=8)

                axs[2].set_ylim(None, max(mean_intensity.max(), avg_intensity.max()) * 1.2)
                axs[2].set_xticks([])
                axs[2].set_xticklabels([])
                axs[2].set_title(f"Mean raw signal for cluster {cluster_num}")
                axs[2].legend(loc="best", handletextpad=0.5, frameon=False, prop={"size": 8})

            else:
                sum_spectrum = self.dataset.spectra_bin if self.dataset.spectra_bin is not None else self.dataset.spectra
                intensity_sum = sum_spectrum.sum().data / sum_spectrum.sum().data.max()
                spectra_profile = self.get_binary_map_spectra_profile(cluster_num)[2]
                intensity = spectra_profile["intensity"].to_numpy() / spectra_profile["intensity"].max()

                axs[2].plot(spectra_profile["energy"], intensity_sum, alpha=1, linewidth=0.7,
                            linestyle="dotted", color=sns.color_palette()[0],
                            label="Normalised sum spectrum")

                axs[2].plot(spectra_profile["energy"], intensity, linewidth=1, color=plot_color)

                axs[2].set_xticks(np.arange(0, 12, step=1))
                axs[2].set_yticks(np.arange(0, 1.1, step=0.2))
                axs[2].set_xticklabels(np.arange(0, 12, step=1).round(1), fontsize=8)
                axs[2].set_yticklabels(np.arange(0, 1.1, step=0.2).round(1), fontsize=8)
                axs[2].set_xlim(spectra_range[0], spectra_range[1])
                axs[2].set_ylim(None, intensity.max() * 1.35)
                axs[2].set_xlabel("Energy / keV", fontsize=10)
                axs[2].set_ylabel("Intensity / a.u.", fontsize=10)
                axs[2].legend(loc="upper right", handletextpad=0.5, frameon=False, prop={"size": 7})

                # Peak annotation with fallback
                energy_array = np.array(spectra_profile["energy"])
                try:
                    zero_energy_idx = np.where(energy_array.round(2) == 0)[0][0]
                except IndexError:
                    zero_energy_idx = 0
                    print("⚠️ Warning: No exact 0.00 keV found in energy axis. Using index 0 for peak lookup.")

                for el in self.dataset.feature_list:
                    if el not in self.peak_dict:
                        continue  # 🚫 skip non-element labels like "Navigator"
                    idx_offset = int(self.peak_dict[el] * 100) + 1
                    try:
                        peak_sum = intensity_sum[zero_energy_idx:][idx_offset]
                        peak_single = intensity[zero_energy_idx:][idx_offset]
                        peak = max(peak_sum, peak_single)
                        axs[2].vlines(self.peak_dict[el], 0, 0.9 * peak, linewidth=0.7,
                                      color="grey", linestyles="dashed")
                        axs[2].text(self.peak_dict[el] - 0.1, peak + (intensity.max() / 20),
                                    el, rotation="vertical", fontsize=7.5)
                    except IndexError:
                        print(f"⚠️ Skipping peak {el}: index {idx_offset} out of range for cluster {cluster_num}")

        except Exception as e:
            axs[2].text(0.5, 0.5, f"Error in spectra plot:\n{str(e)}", ha="center")
            axs[2].set_axis_off()

        fig.subplots_adjust(wspace=0.05, hspace=0.2)
        fig.set_tight_layout(True)
        plt.show()
        return fig

    def plot_phase_map(self, cmap=None, alpha_cluster_map=0.75, phase_override=None):
        import matplotlib.pyplot as plt
        import numpy as np

        # If no cmap is provided, fall back to self.color_palette
        cmap = cmap or self.color_palette

        # Get the base image (navigation / intensity)
        if type(self.dataset) not in [IMAGEDataset, PIXLDataset]:
            img = self.nav_img.data 
        else:
            img = resize(self.dataset.intensity_map, self.dataset.chemical_maps.shape[:2])

        # Use the override phase map if provided, else fall back to self.labels
        phase = phase_override if phase_override is not None else self.labels.reshape(self.height, self.width)

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4), dpi=100)

        axs[0].imshow(img, cmap="gray", interpolation="none")
        axs[0].set_title("Navigation Signal")
        axs[0].axis("off")

        axs[1].imshow(img, cmap="gray", interpolation="none", alpha=1.0)

        # Overlay the cluster map (respects NaNs in phase_override)
        im = axs[1].imshow(
            phase,
            cmap=cmap,
            interpolation="none",
            norm=self.color_norm,
            alpha=alpha_cluster_map,
        )

        axs[1].axis("off")
        axs[1].set_title("Cluster map")

        fig.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.show()
        return fig

    def plot_binary_map_spectra_profile(
        self, cluster_num, normalisation=True, spectra_range=(0, 8), **kwargs
    ):
        # Extract color explicitly
        color = kwargs.pop("color", None)

        binary_map, binary_map_indices, spectra_profile = self.get_binary_map_spectra_profile(
            cluster_num, use_label=True
        )

        if type(self.dataset) not in [IMAGEDataset, PIXLDataset]:
            ncols, figsize, gridspec_kw = 3, (13, 3), {"width_ratios": [1, 1, 2]}
        else:
            ncols, figsize, gridspec_kw = 2, (6, 3), None

        # Only pass valid kwargs to plt.subplots
        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=figsize,
            dpi=96,
            gridspec_kw=gridspec_kw
        )

        c = mcolors.ColorConverter().to_rgb

        if color is not None:
            phase_color = color  # 🎯 use passed-in color
        else:
            phase_color = plt.cm.get_cmap(self.color_palette)(
                cluster_num / (self.n_components - 1)
            )

        if isinstance(phase_color, tuple) or isinstance(phase_color, list):
            # phase_color is already RGB tuple/list, no need to convert
            rgb_color = phase_color
        else:
            # phase_color is probably a string or something else
            rgb_color = c(phase_color)
            
        if np.allclose(rgb_color, (0.0, 0.0, 0.0)):
            rgb_color = (1.0, 1.0, 1.0)  # use white instead of black
            
        if any(v > 1 for v in rgb_color):
            rgb_color = tuple(np.array(rgb_color) / 255)
        cmap = make_colormap([(0.0, c("k")), (1.0, rgb_color)])
        
        if np.sum(binary_map) == 0:
            print(f"⚠️ Cluster {cluster_num} binary map is empty.")

        axs[0].imshow(binary_map, cmap=cmap, vmin=0, vmax=1)
        axs[0].set_title(f"Binary map (cluster {cluster_num})", fontsize=10)
        axs[0].axis("off")
        axs[0].set_aspect("equal", "box")

        # Navigation image logic (unchanged)
        if type(self.dataset) not in [IMAGEDataset, PIXLDataset]:
            nav_img = self.dataset.nav_img_bin.data if self.dataset.nav_img_bin else self.dataset.nav_img.data
        else:
            if self.dataset.intensity_map.shape[:2] != self.dataset.chemical_maps.shape[:2]:
                nav_img = resize(self.dataset.intensity_map, self.dataset.chemical_maps.shape[:2])
            else:
                nav_img = self.dataset.intensity_map
                
            # Match shape if needed
        if nav_img.shape != binary_map.shape:
            try:
                target_shape = binary_map.shape[::-1]  # reverses (y, x) → (x, y)
                nav_img = (self.dataset.nav_img_bin or self.dataset.nav_img).rebin(target_shape).data
            except Exception as e:
                print(f"⚠️ Could not rebin nav_img: {e}")

        axs[1].imshow(nav_img, cmap="gray", interpolation="none", alpha=0.9)
        axs[1].scatter(
            binary_map_indices[1], binary_map_indices[0], c="r", alpha=0.2, s=1.5
        )
        axs[1].grid(False)
        axs[1].axis("off")
        axs[1].set_title("Navigation Signal + Binary Map", fontsize=10)

        if type(self.dataset) not in [IMAGEDataset, PIXLDataset]:
            intensity = spectra_profile["intensity"].to_numpy()
            if normalisation and intensity.max() > 0:
                intensity = intensity / intensity.max()

            energy = spectra_profile["energy"].to_numpy()

            # Use provided color if available
            plot_color = color or (
                plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1)
                if self.n_components <= 10
                else plt.cm.get_cmap(self.color_palette)(cluster_num / (self.n_components - 1))
            )

            axs[2].plot(energy, intensity, linewidth=1, color=plot_color)

            # Plot peak lines
            for el in self.peak_list:
                peak_energy = self.peak_dict.get(el)
                if peak_energy is None:
                    continue

                idxs = np.where(np.isclose(energy, peak_energy, atol=0.01))[0]
                if len(idxs) == 0:
                    continue
                peak_idx = idxs[0]
                peak_value = intensity[peak_idx]

                axs[2].vlines(
                    peak_energy, 0, 0.9 * peak_value, linewidth=0.7, color="grey", linestyles="dashed"
                )
                axs[2].text(
                    peak_energy - 0.1,
                    peak_value + (intensity.max() / 20),
                    el,
                    rotation="vertical",
                    fontsize=8,
                )

            axs[2].set_xticks(np.arange(spectra_range[0], spectra_range[1], step=1))
            axs[2].set_xticklabels(
                np.arange(spectra_range[0], spectra_range[1], step=1), fontsize=8
            )

            if normalisation:
                axs[2].set_yticks(np.arange(0, 1.1, step=0.2))
                axs[2].set_yticklabels(np.arange(0, 1.1, step=0.2).round(1), fontsize=8)
            else:
                try:
                    ymax = intensity.max().round()
                    step = max(1, int(ymax / 5))
                    yticks = np.arange(0, int(ymax) + 1, step=step)
                    axs[2].set_yticks(yticks)
                    axs[2].set_yticklabels(yticks, fontsize=8)
                except ZeroDivisionError:
                    pass

            axs[2].set_xlim(spectra_range[0], spectra_range[1])
            axs[2].set_ylim(None, intensity.max() * 1.2)
            axs[2].set_xlabel("Energy / keV", fontsize=10)
            axs[2].set_ylabel("X-rays / Counts", fontsize=10)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_binary_map(
        self,
        cluster_num,
        binary_filter_args={
            "threshold": 0.8,
            "denoise": False,
            "keep_fraction": 0.13,
            "binary_filter_threshold": 0.2,
        },
        save=None,
        **kwargs,
    ):

        binary_map, binary_map_indices, spectra_profile = self.get_binary_map_spectra_profile(
            cluster_num, **binary_filter_args
        )

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.5), dpi=96, **kwargs)

        axs[0].imshow(binary_map, interpolation="none", alpha=1)
        axs[0].set_title(f"Filtered Binary map (cluster {cluster_num})")
        axs[0].axis("off")
        axs[0].set_aspect("equal", "box")

        axs[1].imshow(self.dataset.nav_img_bin.data, cmap="gray", interpolation="none", alpha=1)
        axs[1].scatter(
            binary_map_indices[1], binary_map_indices[0], c="r", alpha=0.05, s=1.2
        )
        axs[1].grid(False)
        axs[1].axis("off")
        axs[1].set_title(f"Naviation Signal + Phase Map (cluster {cluster_num})")

        # fig.subplots_adjust(left=0.1)
        plt.tight_layout()

        if save is not None:
            fig.savefig(save, bbox_inches="tight", pad_inches=0.02)

        plt.show()

    def plot_unmixed_profile(self, components, peak_list=[]):
        if len(peak_list) == 0:
            peak_list = self.peak_list
        cpnt_num = len(components.columns.to_list())
        if cpnt_num > 4:
            n_rows = (cpnt_num + 3) // 4
            n_cols = 4
        else:
            n_rows = 1
            n_cols = cpnt_num

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 2.6), dpi=150
        )
        for row in range(n_rows):
            for col in range(n_cols):
                cur_cpnt = (row * n_cols) + col
                if cur_cpnt > cpnt_num - 1:  # delete the extra subfigures
                    fig.delaxes(axs[row, col])
                else:
                    cpnt = f"cpnt_{cur_cpnt}"
                    if cpnt_num > 4:
                        axs_sub = axs[row, col]
                    else:
                        axs_sub = axs[col]

                    if type(self.dataset) in [IMAGEDataset, PIXLDataset]:
                        axs_sub.bar(
                            self.dataset.feature_list,
                            components[cpnt],
                            width=0.6,
                            linewidth=1,
                        )
                        for i in range(len(self.dataset.feature_list)):
                            y = components[cpnt][i] + components[cpnt].max()*0.03
                            axs_sub.text(i-len(self.dataset.feature_list[i])*0.11,y,self.dataset.feature_list[i], fontsize=8)
                            
                        axs_sub.set_ylim(None, components[cpnt].max()*1.2)
                        axs_sub.set_xticks([])
                        axs_sub.set_xticklabels([])
                        axs_sub.set_title(f"cpnt_{cur_cpnt}")
                    
                    else:
                        axs_sub.plot(self.energy_axis, components[cpnt], linewidth=1)
                        axs_sub.set_xlim(0, 8)
                        axs_sub.set_ylim(None, components[cpnt].max() * 1.3)
                        axs_sub.set_ylabel("Intensity")
                        axs_sub.set_xlabel("Energy (keV)")
                        axs_sub.set_title(f"cpnt_{cur_cpnt}")
    
                        if np.array(self.energy_axis).min() <= 0.0:
                            zero_energy_idx = np.where(
                                np.array(self.energy_axis).round(2) == 0
                            )[0][0]
                        else:
                            zero_energy_idx = 0
                        intensity = components[cpnt].to_numpy()
                        for el in peak_list:
                            if el not in self.peak_dict:
                                continue  # Skip labels like "Navigator"
                            try:
                                idx_offset = int(self.peak_dict[el] * 100) + 1
                                peak = intensity[zero_energy_idx:][idx_offset]
                                axs_sub.vlines(
                                    self.peak_dict[el],
                                    0,
                                    0.9 * peak,
                                    linewidth=1,
                                    color="grey",
                                    linestyles="dashed",
                                )
                                axs_sub.text(
                                    self.peak_dict[el] - 0.18,
                                    peak + (intensity.max() / 15),
                                    el,
                                    rotation="vertical",
                                    fontsize=8,
                                )
                            except IndexError:
                                print(f"⚠️ Skipping peak {el}: index {idx_offset} out of range in component {cpnt}")


        fig.subplots_adjust(hspace=0.3, wspace=0.0)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_spectra_profile(self, cluster_num, peak_list, binary_filter_args):
        spectra_profile = self.get_binary_map_spectra_profile(
            cluster_num, **binary_filter_args
        )[2]
        intensity = spectra_profile["intensity"].to_numpy()

        fig, axs = plt.subplots(1, 1, figsize=(4, 2), dpi=150)
        axs.set_xticks(np.arange(0, 12, step=1))
        axs.set_yticks(
            np.arange(0, int(intensity.max()) + 1, step=int((intensity.max() / 5)))
        )

        axs.set_xticklabels(np.arange(0, 12, step=1), fontsize=8)
        axs.set_yticklabels(
            np.arange(0, int(intensity.max()) + 1, step=int((intensity.max() / 5))),
            fontsize=8,
        )

        axs.set_xlim(0, 8)
        axs.set_ylim(None, intensity.max() * 1.25)
        axs.set_xlabel("Energy axis / keV", fontsize=10)
        axs.set_ylabel("X-rays / Counts", fontsize=10)

        if self.n_components <= 10:
            axs.plot(
                spectra_profile["energy"],
                spectra_profile["intensity"],
                linewidth=1,
                color=plt.cm.get_cmap(self.color_palette)(cluster_num * 0.1),
            )
        else:
            axs.plot(
                spectra_profile["energy"],
                spectra_profile["intensity"],
                linewidth=1,
                color=plt.cm.get_cmap(self.color_palette)(
                    cluster_num * (self.n_components - 1) ** -1
                ),
            )

        zero_energy_idx = np.where(np.array(spectra_profile["energy"]).round(2) == 0)[0][0]
        for el in peak_list:
            peak = intensity[zero_energy_idx:][int(self.peak_dict[el] * 100) + 1]
            axs.vlines(
                self.peak_dict[el],
                0,
                int(0.9 * peak),
                linewidth=0.7,
                color="grey",
                linestyles="dashed",
            )
            axs.text(
                self.peak_dict[el] - 0.075,
                peak + (int(intensity.max()) / 20),
                el,
                rotation="vertical",
                fontsize=7.5,
            )
        plt.show()

    def plot_ternary_composition(self, **kwargs): # see args for cluster_quantification
        cluster_element_intensities = self.cluster_quantification(**kwargs)
        
        fig = go.Figure(px.scatter_ternary(cluster_element_intensities, 
                                   *cluster_element_intensities.columns,
                                   template='none',
                                   opacity=0.5)
        )

        fig.update_layout(title="Ternary diagram in at.%",
                          title_x=0.5,
                          width=500,
                          height=500)
        
        fig.update_traces(marker=dict(size=3.0,
                                      line=dict(width=0)),
                          selector=dict(mode='markers'),
                         )
        fig.show()

    ######################
    # SIGMA2 imporvements#
    ######################
    
    def perform_NMF(self, **kwargs):
        """
        Perform Non-negative Matrix Factorisation on the spectra in the clusters in the PixelSegmentor Object.
    
        Add these weights and components to the PixelSegmentor object under self.NMF_weights and self.NMF_components
    
        Parameters
        ----------
        **kwargs - List of arguments to be input to the 'get_unmixed_spectra_profile' method
        """
    
        self.NMF_weights,self.NMF_components=self.get_unmixed_spectra_profile(**kwargs)




    def plot_NMF_map(self,weights, cmap=None, alpha_cluster_map=0.75):
        """
        Plot the NMF components alongside the navigation image.

        Parameters
        ----------
        cmap              : str
                            suitable matplotlib string to dfeine a colormap
                   
        alpha_cluster_map : float
                            alpha value for the plot of the NMF clusters map overlayed on the navigation image. Default 0.75
                             


        """
        cmap = self.color_palette if cmap is None else cmap
        if type(self.dataset) not in [IMAGEDataset, PIXLDataset]:
            img = self.nav_img.data 
        else:
            img = resize(self.dataset.intensity_map, self.dataset.chemical_maps.shape[:2])

        shape = self.get_binary_map_spectra_profile(0)[0].shape
        phase_img = np.zeros((shape[0], shape[1], self.n_components))

        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4), dpi=100)
        axs[1].imshow(img, cmap="gray", interpolation="none", alpha=1.0) #showing the navigation image

        

        cpnt_names = [f'cluster_{i}' for i in range(len(weights.columns))] 
        cpnt_options = [x for x in zip(cpnt_names, weights.columns)]


        #showing the navigation image
        axs[0].imshow(img, cmap="gray", interpolation="none")
        axs[0].set_title("Navigation Signal")
        tmp_cumulative = np.zeros(shape)

        #need the inverse of the components matrix
        for i, phase in enumerate(weights.columns):
            print(i)
            if phase!='None':
                cpnt_weights = weights[phase]/weights[phase].max()
                tmp = np.zeros(shape)
                for j in range(self.n_components):
                    try:
                        idx = self.get_binary_map_spectra_profile(j)[1]
                        tmp[idx] = cpnt_weights[j]
                    except ValueError:
                        pass
                        # print(f'warning: no pixel is assigned to cpnt_{j}.')
                phase_img[:, :, i] = tmp
            else:
                phase_img[:, :, i] = np.zeros(shape)
            




        for i in range(len(weights.columns)):
            if len(weights) <= 10:
                axs[1].imshow(
                    phase_img[:,:,i],
                    cmap=self.color_palette,
                    interpolation="none",
                    norm=self.color_norm,
                    alpha=alpha_cluster_map,
                )
            else:
                axs[1].imshow(
                    phase_img[:,:,i],
                    cmap=self.color_palette,
                    interpolation="none",
                    alpha=alpha_cluster_map,
                    norm=self.color_norm,
                )
        axs[1].axis("off")
        axs[1].set_title("NMF components map")

        fig.subplots_adjust(wspace=0.05, hspace=0.0)
        plt.show()
        return fig
        
    def save_state(self, filepath):
        """Save the current latent space, labels, and cluster colors to a file."""
        state = {
            'latent': self.latent,
            'labels': self.labels,
            'cluster_colors': self.cluster_colors,
            'color_palette': self.color_palette,
            'n_components': self.n_components,
            'height': self.height,
            'width': self.width,
            "method": self.method,
            "method_args": self.method_args,
            'peak_dict': self.peak_dict,
            'manual_cluster_colors': self.manual_cluster_colors,

        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"✅ Saved state to {filepath}")

    def load_state(self, filepath):
        """Load latent space, labels, and cluster colors from a file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.latent = state['latent']
        self.labels = state['labels']
        self.cluster_colors = state['cluster_colors']
        self.color_palette = state['color_palette']
        self.n_components = state['n_components']
        self.height = state['height']
        self.width = state['width']
        self.method = saved_data['method']
        self.method_args = saved_data['method_args']
        self.peak_dict = saved_data['peak_dict']
        self.manual_cluster_colors = saved_data['manual_cluster_colors']
        print(f"✅ Loaded state from {filepath}")    
        
        
     
     
    
    def restore_dataset_state(self):
        if not hasattr(self, "dataset"):
            raise AttributeError("Cannot restore state: 'dataset' is not set.")

        self.dataset_norm = self.dataset.normalised_elemental_data
        self.height = self.dataset_norm.shape[0]
        self.width = self.dataset_norm.shape[1]

        if hasattr(self.dataset, "spectra_bin") and self.dataset.spectra_bin is not None:
            self.spectra = self.dataset.spectra_bin
        else:
            self.spectra = getattr(self.dataset, "spectra", None)

        if hasattr(self.dataset, "nav_img_bin") and self.dataset.nav_img_bin is not None:
            self.nav_img = self.dataset.nav_img_bin
        else:
            self.nav_img = getattr(self.dataset, "nav_img", None)

        if hasattr(self.dataset, "spectra") and self.spectra is not None:
            ax = self.spectra.axes_manager[2]
            self.energy_axis = [(a * ax.scale + ax.offset) for a in range(ax.size)]

        self.peak_list = getattr(self.dataset, "feature_list", [])


    @classmethod
    def from_saved_state(cls, filepath, dataset=None):
        """Instantiate a PixelSegmenter from saved state. Dataset must be provided manually if needed."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls.__new__(cls)

        # Set attributes from saved state
        instance.latent = state['latent']
        instance.labels = state['labels']
        instance.cluster_colors = state['cluster_colors']
        instance.color_palette = state['color_palette']
        instance.n_components = state['n_components']
        instance.height = state['height']
        instance.width = state['width']
        instance.method= state["method"]
        instance.method_args = state["method_args"]
        instance.peak_dict = state["peak_dict"]
        instance.manual_cluster_colors = state['manual_cluster_colors']

        if not hasattr(instance, 'color_norm'):
            instance.color_norm = mcolors.Normalize()
        # You can choose to warn or error if dataset is not passed
        if dataset is None:
            print("⚠️ Dataset not provided. You must set 'ps.data' manually before using this instance.")
        else:
            instance.dataset = dataset
            instance.restore_dataset_state()

        return instance
        
    def export_cluster_masks(
        self,
        output_dir="cluster_masks",
        high_res_path=None,
        upscale_interpolation="nearest"
    ):
        """
        Export binary mask images (black/white) for each cluster.

        Parameters:
        -----------
        output_dir : str
            Directory to save output mask .tif files.
        high_res_path : str or None
            Optional path to a high resolution reference image.
            If provided, masks will be upscaled to match this image's resolution.
        upscale_interpolation : str
            Interpolation method used for resizing (e.g. "nearest", "bilinear", "bicubic").
        """
        os.makedirs(output_dir, exist_ok=True)

        label_map = self.labels.reshape(self.height, self.width)
        cluster_ids = sorted([c for c in np.unique(label_map) if c != -1])

        # Get target resolution
        if high_res_path:
            import imageio
            ref_img = imageio.imread(high_res_path)
            target_shape = ref_img.shape[:2]
        else:
            target_shape = self.nav_img.data.shape[:2]

        for cluster_id in cluster_ids:
            # Binary mask: 1 where cluster matches, else 0
            binary_mask = (label_map == cluster_id).astype(np.uint8)

            # Resize if needed
            if binary_mask.shape != target_shape:
                binary_mask = resize(
                    binary_mask,
                    output_shape=target_shape,
                    order=0 if upscale_interpolation == "nearest" else 1,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.uint8)

            # Save binary mask as .tif (0 for background, 255 for foreground)
            out_path = os.path.join(output_dir, f"cluster_{cluster_id}.tif")
            imsave(out_path, binary_mask * 255)

        print(f"Exported {len(cluster_ids)} cluster mask(s) to '{output_dir}'.")
        
        
    def get_unmixed_spectra_profile_init_guess(
        self,
        clusters_to_be_calculated="All",
        n_components="All",
        normalised=True,
        method="NMF",
        method_args=None,
        seed_clusters=None,      # NEW: list/tuple of ints (e.g. [0,3,5]) or strings like "cluster_3"
        seed_endmembers=None,    # NEW: array-like (k, n_features) to seed H directly
        fill_missing_with="nndsvda",  # how to fill remaining components if seeds < k
    ):
        assert method == "NMF", "Only NMF is supported currently."
        from sklearn.decomposition import NMF
        import numpy as np
        import pandas as pd

        method_args = {} if method_args is None else dict(method_args)

        # --- Data: X = clusters x features
        # get_all_spectra_profile returns (array(C x F), cluster_ids)
        spectra_result = self.get_all_spectra_profile(normalised)

        # accept either (arr, cluster_ids) or a bare array/df for backwards compatibility
        if isinstance(spectra_result, tuple):
            spectra_arr, cluster_ids = spectra_result[0], spectra_result[1]
        else:
            spectra_arr = spectra_result
            # if cluster ids not provided, fallback to integer indices 0..C-1
            cluster_ids = np.arange(spectra_arr.shape[0])

        # Convert to numpy array if needed and validate
        if isinstance(spectra_arr, pd.DataFrame):
            arr = spectra_arr.values
        else:
            arr = np.asarray(spectra_arr, dtype=float)

        if arr.ndim != 2:
            raise ValueError(f"Expected spectra_profiles to be 2D (C x F), got shape {arr.shape}")

        # Build DataFrame: features x clusters with cluster_ids as columns
        spectra_profiles_ = pd.DataFrame(arr.T, columns=list(cluster_ids))  # (F x C)


        # Optional subset of clusters to process
        if clusters_to_be_calculated != "All":
            spectra_profiles_ = spectra_profiles_[clusters_to_be_calculated]

        # Number of NMF components
        if n_components == "All":
            n_components = spectra_profiles_.shape[1]
        k = int(n_components)

        # X is samples x features = clusters x features
        X = spectra_profiles_.to_numpy().T
        n_samples, n_features = X.shape

        # ---- Build custom init if requested
        W0 = H0 = None
        use_custom = (seed_clusters is not None) or (seed_endmembers is not None)
        if use_custom:
            # Build H0 (k x n_features)
            if seed_endmembers is not None:
                H0 = np.asarray(seed_endmembers, dtype=float)
                if H0.ndim != 2 or H0.shape[1] != n_features:
                    raise ValueError(f"seed_endmembers must be (k, n_features={n_features}), got {H0.shape}")
                if np.any(H0 < 0):
                    raise ValueError("seed_endmembers must be nonnegative.")
            else:
                # seed from cluster indices found in current subset columns
                cols = list(spectra_profiles_.columns)
                # normalise/parse possible "cluster_3" strings to ints
                seeds_idx = []
                for s in seed_clusters:
                    if isinstance(s, str) and s.startswith("cluster_"):
                        s = int(s.split("_", 1)[1])
                    s = int(s)
                    if s not in cols:
                        raise ValueError(f"Seed cluster {s} is not in current selection {cols}.")
                    seeds_idx.append(cols.index(s))  # position among current columns

                # rows of H0 are the chosen cluster spectra
                H0 = X[seeds_idx, :]  # (#seeds x n_features)

            # if fewer than k seeds, fill the rest
            if H0.shape[0] < k:
                r = k - H0.shape[0]
                if fill_missing_with in ("nndsvd", "nndsvda", "nndsvdar"):
                    filler_model = NMF(n_components=k, init=fill_missing_with, random_state=method_args.get("random_state", 0))
                    _ = filler_model.fit_transform(X)
                    Hfill = filler_model.components_  # (k x F)
                    # take r rows that we don't already have (simple pick from the end)
                    H0 = np.vstack([H0, Hfill[-r:, :]])
                elif fill_missing_with == "zeros":
                    H0 = np.vstack([H0, np.full((r, n_features), 1e-6)])
                else:
                    raise ValueError("fill_missing_with must be one of {'nndsvd','nndsvda','nndsvdar','zeros'}")

            # if more than k seeds, trim
            if H0.shape[0] > k:
                H0 = H0[:k, :]

            # light normalization to avoid scale pathologies
            H0 = H0 / np.maximum(np.linalg.norm(H0, axis=1, keepdims=True), 1e-12)

            # crude nonnegative W0 consistent with H0
            W0 = np.maximum(X @ H0.T, 1e-12)
            method_args["init"] = "custom"
        else:
            method_args.setdefault("init", "nndsvda")

        # --- Fit NMF
        model = NMF(n_components=k, **method_args)
        if use_custom:
            W = model.fit_transform(X, W=W0, H=H0)
        else:
            W = model.fit_transform(X)
        H = model.components_  # (k x n_features)

        self.NMF_recon_error = model.reconstruction_err_

        # --- Tidy outputs
        weights = pd.DataFrame(
            W.round(3),
            columns=[f"w_{i}" for i in range(k)],
            index=[f"cluster_{c}" for c in spectra_profiles_.columns],
        )
        components = pd.DataFrame(
            H.T.round(3),
            columns=[f"cpnt_{i}" for i in range(k)],
        )
        return weights, components
        
    def compute_cluster_proportions(
        self,
        use_soft: bool = False,
        soft_threshold: float = None,
        exclude_noise: bool = True,
        noise_label: int = -1,
        area_units: str = "pixels",  # "pixels" (default) or "um2"
        return_counts: bool = True,   # now True by default so pixel_count is returned
        plot: bool = False,
    ):
        """
        Compute pixel proportions for each cluster and optionally plot them.

        Returns a pandas.DataFrame indexed by cluster_id with columns:
          - pixel_count (absolute pixel counts or summed probabilities)
          - proportion (fraction of denominator)
          - area_um2 (if calibration available, else NaN)

        Behavior notes
        --------------
        - If `use_soft=True` and self.prob_map exists, pixel_count is the summed
          probabilities for each cluster. Denominator is total pixels unless
          soft_threshold is provided (then only pixels with max prob >= threshold
          are counted in denominator).
        - If `use_soft=False` hard labels (self.labels) are used.
        """
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt
        import matplotlib.ticker as mtick

        # determine spatial size
        if hasattr(self, "height") and hasattr(self, "width"):
            total_pixels = int(self.height) * int(self.width)
        else:
            try:
                sp_shape = self.spectra.data.shape[:2]
                total_pixels = int(sp_shape[0]) * int(sp_shape[1])
            except Exception:
                raise ValueError("Cannot determine spatial dimensions (height/width or spectra).")

        # default empty df
        df = None

        # --- Soft-probability path ---
        if use_soft and hasattr(self, "prob_map") and self.prob_map is not None:
            prob_map = _np.asarray(self.prob_map)  # (n_pixels, n_clusters)
            if prob_map.ndim != 2:
                raise ValueError("prob_map must be 2D (n_pixels x n_components).")
            n_pixels, n_clusters = prob_map.shape
            if n_pixels != total_pixels:
                raise ValueError(f"Mismatch: prob_map has {n_pixels} pixels but expected {total_pixels}.")

            # summed probabilities per cluster
            cluster_sums = prob_map.sum(axis=0).astype(float)

            # denominator handling
            if soft_threshold is not None:
                max_prob = prob_map.max(axis=1)
                include_mask = max_prob >= float(soft_threshold)
                denom = include_mask.sum()
                if denom == 0:
                    raise ValueError("soft_threshold excluded all pixels (denominator 0).")
                cluster_sums = prob_map[include_mask].sum(axis=0).astype(float)
            else:
                denom = n_pixels

            cluster_ids = list(range(n_clusters))
            df = _pd.DataFrame({"pixel_count": cluster_sums}, index=cluster_ids)
            df["proportion"] = df["pixel_count"] / float(denom)

        # --- Hard-label path ---
        else:
            if not hasattr(self, "labels"):
                if hasattr(self, "model") and hasattr(self.model, "predict"):
                    labels = _np.asarray(self.model.predict(self.latent))
                else:
                    raise ValueError("No hard labels found and cannot predict labels.")
            else:
                labels = _np.asarray(self.labels).astype(int)

            if labels.size != total_pixels:
                raise ValueError(f"Mismatch: labels length ({labels.size}) != total pixels ({total_pixels}).")

            unique, counts = _np.unique(labels, return_counts=True)
            df = _pd.DataFrame({"pixel_count": counts}, index=unique.astype(int))

            # drop noise label if requested
            if exclude_noise and (noise_label in df.index):
                df = df.drop(noise_label)

            denom = df["pixel_count"].sum()
            if denom == 0:
                raise ValueError("No pixels in selected clusters (denominator 0).")
            df["proportion"] = df["pixel_count"].astype(float) / float(denom)

        # Ensure index name and sorting
        df.index.name = "cluster_id"
        df = df.sort_index()

        # Compute area_um2 where possible (add column even if NaN)
        area_um2 = _np.nan
        try:
            # prefer self.spectra.axes_manager, fallback to dataset.spectra.axes_manager
            if hasattr(self, "spectra") and hasattr(self.spectra, "axes_manager"):
                pixel_to_um = float(self.spectra.axes_manager[0].scale)
            elif hasattr(self, "dataset") and hasattr(self.dataset, "spectra") and hasattr(self.dataset.spectra, "axes_manager"):
                pixel_to_um = float(self.dataset.spectra.axes_manager[0].scale)
            else:
                pixel_to_um = None

            if pixel_to_um is not None:
                df["area_um2"] = df["pixel_count"] * (pixel_to_um ** 2)
            else:
                df["area_um2"] = _np.nan
        except Exception:
            df["area_um2"] = _np.nan

        # If user doesn't want counts, drop pixel_count but keep area_um2 and proportion
        if not return_counts:
            df = df[["proportion", "area_um2"]]

        # --- Plotting: autoscaled and percentage y-axis ---
        if plot:
            try:
                # Print the table for quick inspection
                print(df.round(4))

                fig, ax = _plt.subplots(figsize=(6, 3.5), dpi=100)

                # bar of proportions
                xs = df.index.astype(str)
                ys = df["proportion"].to_numpy(dtype=float)
                bars = ax.bar(xs, ys, alpha=0.85)

                # y-axis autoscale with margin (don't force 0-1)
                y_max = max(ys.max(), 1e-6)
                top = min(1.0, y_max * 1.25) if y_max > 0 else 0.05
                ax.set_ylim(0, top)

                # Show percentage ticks on y-axis
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                ax.set_ylabel("Proportion (%)")
                ax.set_xlabel("Cluster id")
                ax.set_title("Cluster proportions")

                # annotate small bars with percent labels if they are small
                for rect, val in zip(bars, ys):
                    h = rect.get_height()
                    label = f"{val*100:.2f}%"
                    # place label above bar, but adjust if near top
                    ax.annotate(label, xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

                ax.grid(axis="y", linestyle="--", alpha=0.3)
                _plt.tight_layout()
                _plt.show()

            except Exception as e:
                print(f"⚠️ Could not create plot: {e}")

        return df

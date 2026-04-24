from umap import UMAP # for UMAP latent space projections
import sys # for relative imports of sigma

sys.path.insert(0,"..")



from sigma.utils import normalisation as norm 
from sigma.utils import visualisation as visual
from sigma.utils.load import SEMDataset
from sigma.src.utils import same_seeds
from sigma.src.dim_reduction import Experiment
from sigma.models.autoencoder import AutoEncoder
from sigma.src.segmentation import PixelSegmenter
from sigma.gui import gui
from sigma.utils.loadtem import TEMDataset


def extract_cluster_spectra(self, cluster_ids="All", normalised=True, return_hyperspy=False):
    """
    Extract averaged spectra (and feature vector) for each cluster.

    Parameters
    ----------
    cluster_ids : "All" or int or iterable of ints
        Which cluster ids to extract. If "All", uses unique labels found in self.labels (excludes -1).
    normalised : bool
        If True, return intensities normalised to max=1 per cluster.
    return_hyperspy : bool
        If True, create a hyperspy.Signal1D for each cluster's mean spectrum.

    Returns
    -------
    dict
        {
            cluster_id: {
                "spectra_profile": pandas.DataFrame with columns ["energy", "intensity"],
                "mean_spectrum": np.ndarray (1D),
                "energy": np.ndarray (1D),
                "feature_vector": np.ndarray or None,
                "hyperspy": hs.signals.Signal1D or None,
                "binary_map": np.ndarray,
                "binary_map_indices": tuple
            },
            ...
        }
    """
    import numpy as _np
    import pandas as _pd
    import hyperspy.api as _hs

    if not hasattr(self, "labels"):
        raise ValueError("No labels available on this PixelSegmenter instance.")

    if not hasattr(self, "spectra") or self.spectra is None:
        raise ValueError("self.spectra is missing.")

    if not hasattr(self.spectra, "data") or self.spectra.data is None:
        raise ValueError("self.spectra.data is missing.")

    if self.spectra.data.ndim != 3:
        raise ValueError(
            f"Expected self.spectra.data to be 3D (x, y, energy), got shape {self.spectra.data.shape}."
        )

    labels_arr = _np.asarray(self.labels).astype(int)
    all_clusters = _np.unique(labels_arr)
    all_clusters = all_clusters[all_clusters >= 0]  # skip noise / outliers

    if cluster_ids == "All":
        clusters_to_iter = list(all_clusters)
    else:
        if _np.isscalar(cluster_ids):
            cluster_ids = [cluster_ids]
        cluster_ids = _np.asarray(list(cluster_ids), dtype=int)
        clusters_to_iter = [c for c in cluster_ids if c in all_clusters]
        if len(clusters_to_iter) == 0:
            raise ValueError("No requested cluster ids are present in labels.")

    out = {}

    for cid in clusters_to_iter:
        cid = int(cid)

        try:
            binary_map, binary_map_indices, spectra_profile = self.get_binary_map_spectra_profile(
                cluster_num=cid,
                use_label=True
            )
        except Exception as e:
            print(f"⚠️ Skipping cluster {cid}: {e}")
            out[cid] = {"spectra_profile": None, "_error": str(e)}
            continue

        # Fallback: if helper did not return a spectra_profile, compute it directly
        if spectra_profile is None:
            try:
                x_idx, y_idx = binary_map_indices
                if len(x_idx) == 0:
                    raise ValueError(f"Cluster {cid} has no pixels.")

                spectra = _np.array([self.spectra.data[x, y, :] for x, y in zip(x_idx, y_idx)], dtype=float)
                intensity = spectra.mean(axis=0)

                if hasattr(self, "energy_axis") and self.energy_axis is not None:
                    energy = _np.asarray(self.energy_axis, dtype=float)
                else:
                    ax = self.spectra.axes_manager[2]
                    energy = _np.asarray([(a * ax.scale + ax.offset) for a in range(ax.size)], dtype=float)

                spectra_profile = _pd.DataFrame(
                    data=_np.column_stack([energy, intensity]),
                    columns=["energy", "intensity"]
                )
            except Exception as e:
                print(f"⚠️ Could not build spectra_profile for cluster {cid}: {e}")
                out[cid] = {
                    "spectra_profile": None,
                    "_error": str(e),
                    "binary_map": binary_map,
                    "binary_map_indices": binary_map_indices,
                }
                continue

        energy = spectra_profile["energy"].to_numpy()
        intensity = spectra_profile["intensity"].to_numpy(dtype=float)

        if normalised and intensity.max() > 0:
            intensity = intensity / float(intensity.max())
            spectra_profile = _pd.DataFrame(
                data=_np.column_stack([energy, intensity]),
                columns=["energy", "intensity"]
            )

        feature_vector = None
        if hasattr(self, "mu") and self.mu is not None:
            try:
                if hasattr(self.mu, "__len__") and 0 <= cid < len(self.mu):
                    feature_vector = _np.asarray(self.mu[cid])
            except Exception:
                feature_vector = None

        hs_signal = None
        if return_hyperspy:
            try:
                hs_signal = _hs.signals.Signal1D(intensity)
                try:
                    hs_signal.metadata.General.title = f"cluster_{cid}_mean_spectrum"
                except Exception:
                    pass

                try:
                    energy_arr = _np.asarray(energy, dtype=float)
                    if energy_arr.ndim == 1 and energy_arr.size > 1:
                        step = float(_np.median(_np.diff(energy_arr)))
                        hs_signal.axes_manager[0].scale = step
                        hs_signal.axes_manager[0].offset = float(energy_arr[0])
                        hs_signal.axes_manager[0].name = "Energy"
                        hs_signal.axes_manager[0].units = "keV"
                except Exception:
                    pass

            except Exception as e:
                print(f"⚠️ Could not create hyperspy signal for cluster {cid}: {e}")
                hs_signal = None

        out[cid] = {
            "spectra_profile": spectra_profile,
            "mean_spectrum": intensity,
            "energy": energy,
            "feature_vector": feature_vector,
            "hyperspy": hs_signal,
            "binary_map": binary_map,
            "binary_map_indices": binary_map_indices,
        }

    return out
    
from typing import Iterable, Dict, Any
import os
import gc
import traceback
import numpy as np

def batch_process_and_extract(
    inputs: Iterable,
    input_type: str = "path",                  # "path" or "dataset"
    preprocessing_args: Dict[str, Any] = None,
    projection: str = "umap",                  # "umap" or "autoencoder"
    projection_args: Dict[str, Any] = None,
    clustering: str = "GaussianMixture",       # "BayesianGaussianMixture", "GaussianMixture", "HDBSCAN"
    clustering_args: Dict[str, Any] = None,
    extractor_kwargs: Dict[str, Any] = None,   # passed to extract_cluster_spectra()
    pixelsegmenter_class=None,                 # defaults to PixelSegmenter if None
    verbose: bool = True,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Batch process many files/datasets:
      - preprocessing (bin/smooth/normalise)
      - projection (UMAP or autoencoder)
      - clustering
      - extraction of averaged spectra per cluster

    Memory optimisations:
      - only stores minimal outputs in `results`
      - clears large intermediate objects after each file
      - calls gc.collect() after each iteration
      - stores label maps as int16 where possible
    """

    def _as_file_list(x):
        if isinstance(x, str):
            if os.path.isdir(x):
                return [os.path.join(x, f) for f in os.listdir(x)]
            return [x]
        return list(x)

    def _norm_setting_from_methods(norm_methods):
        if len(norm_methods) == 3:
            return [norm.neighbour_averaging, norm.zscore, norm.softmax]
        if len(norm_methods) == 2:
            if "softmax" not in norm_methods:
                return [norm.neighbour_averaging, norm.zscore]
            if "zscore" not in norm_methods:
                return [norm.neighbour_averaging, norm.softmax]
            return [norm.zscore, norm.softmax]
        if len(norm_methods) == 1:
            if "softmax" in norm_methods:
                return [norm.softmax]
            if "zscore" in norm_methods:
                return [norm.zscore]
            return [norm.neighbour_averaging]
        raise ValueError(f"norm_methods must have length 1, 2, or 3. Got {norm_methods!r}")

    files = _as_file_list(inputs)

    preprocessing_args = {} if preprocessing_args is None else dict(preprocessing_args)
    projection_args = {} if projection_args is None else dict(projection_args)
    clustering_args = {} if clustering_args is None else dict(clustering_args)
    extractor_kwargs = {} if extractor_kwargs is None else dict(extractor_kwargs)

    if pixelsegmenter_class is None:
        pixelsegmenter_class = PixelSegmenter

    results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for i, item in enumerate(files):
        file_key = f"item_{i}"

        dataset = None
        sem = None
        ps = None
        latent = None
        data = None
        spectra_for_extraction = None
        nav_img_for_extraction = None
        label_map = None
        umap = None
        ex = None

        try:
            if input_type == "path":
                dataset = SEMDataset(item)
                file_key = item
            elif input_type == "dataset":
                dataset = item
                if hasattr(dataset, "name"):
                    file_key = str(dataset.name)
            else:
                raise ValueError("input_type must be 'path' or 'dataset'")

            if verbose:
                print(f"\n➡️ Processing '{file_key}'")

            sem = dataset

            if preprocessing_args.get("xray_lines") is not None:
                sem.set_feature_list(preprocessing_args["xray_lines"])

            if preprocessing_args.get("rebin", False):
                if verbose:
                    print(f"  - Rebinning: {preprocessing_args.get('bin_size')}")
                sem.rebin_signal(size=preprocessing_args["bin_size"])

            if preprocessing_args.get("peak_int_norm", False):
                if verbose:
                    print("  - Peak intensity normalisation")
                sem.peak_intensity_normalisation()

            if preprocessing_args.get("remove_first_peak", False):
                if verbose:
                    print(f"  - Removing first peak up to {preprocessing_args.get('first_peak_end')}")
                sem.remove_first_peak(end=preprocessing_args["first_peak_end"])

            norm_methods = preprocessing_args.get("norm_methods", ())
            if norm_methods:
                norm_setting = _norm_setting_from_methods(norm_methods)
                if verbose:
                    print(f"  - Normalising with: {norm_methods}")
                sem.normalisation(norm_setting)

            if preprocessing_args.get("use_nav_img", False):
                norm_setting = _norm_setting_from_methods(preprocessing_args.get("norm_methods", ()))
                sem.get_feature_maps_with_nav_img(normalisation=norm_setting)

            if verbose:
                print(f"File {i}: Pre-processing complete")

            spectra_for_extraction = (
                sem.spectra_bin if getattr(sem, "spectra_bin", None) is not None else sem.spectra
            )
            nav_img_for_extraction = (
                sem.nav_img_bin if getattr(sem, "nav_img_bin", None) is not None else sem.nav_img
            )

            if projection.lower() == "autoencoder":
                if verbose:
                    print(f"performing autoencoder projection on file {i}")

                seed_no = projection_args["seed_no"]
                same_seeds(seed_no)

                model_name = projection_args["model_name"]
                autoencoder_args = projection_args["autoencoder_args"]

                general_results_dir = "./"
                ex = Experiment(
                    descriptor=model_name,
                    general_results_dir=general_results_dir,
                    model=AutoEncoder,
                    model_args=autoencoder_args,
                    chosen_dataset=sem.normalised_elemental_data,
                    save_model_every_epoch=True,
                )

                num_epochs = projection_args["num_epochs"]
                patience = projection_args["patience"]
                batch_size = projection_args["batch_size"]
                learning_rate = projection_args["learning_rate"]
                weight_decay = projection_args["weight_decay"]
                task = projection_args["task"]
                noise_added = projection_args["noise_added"]
                KLD_lambda = projection_args["KLD_lambda"]
                criterion = projection_args["criterion"]
                lr_scheduler_args = projection_args["lr_scheduler_args"]

                ex.run_model(
                    num_epochs=num_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    task=task,
                    noise_added=noise_added,
                    KLD_lambda=KLD_lambda,
                    criterion=criterion,
                    lr_scheduler_args=lr_scheduler_args,
                )

                latent = ex.get_latent()

            elif projection.lower() == "umap":
                if verbose:
                    print(f"performing umap projection on file {i}")

                data = sem.normalised_elemental_data.reshape(-1, len(sem.feature_list))
                umap = UMAP(
                    n_neighbors=projection_args["n_neighbours"],
                    min_dist=projection_args["min_dist"],
                    n_components=projection_args["n_components"],
                    metric=projection_args["metric"],
                )
                latent = umap.fit_transform(data)

            else:
                raise ValueError("projection must be 'umap' or 'autoencoder'")

            if latent is None:
                raise ValueError("Projection block did not produce `latent`.")

            method_name = clustering
            method_args = dict(clustering_args)

            if verbose:
                print("Performing Clustering")

            ps = pixelsegmenter_class(
                latent=latent,
                dataset=sem,
                method=method_name,
                method_args=method_args,
            )

            ps.spectra = spectra_for_extraction
            ps.nav_img = nav_img_for_extraction
            ps.dataset.spectra = spectra_for_extraction
            ps.dataset.nav_img = nav_img_for_extraction

            try:
                if hasattr(ps.spectra, "axes_manager") and len(ps.spectra.axes_manager) >= 3:
                    ax = ps.spectra.axes_manager[2]
                    ps.energy_axis = [(a * ax.scale + ax.offset) for a in range(ax.size)]
            except Exception:
                pass

            if verbose:
                print("Clustering Successful")

            if verbose:
                print("labels shape:", np.asarray(ps.labels).shape)
                print("spectra shape:", ps.spectra.data.shape[:2])
                print("dataset_norm shape:", ps.dataset_norm.shape[:2])
                print("unique labels:", np.unique(ps.labels, return_counts=True))

            spectra_dict = extract_cluster_spectra(
                ps,
                cluster_ids="All",
                **extractor_kwargs,
            )

            file_result: Dict[int, Dict[str, Any]] = {}
            for cid, d in spectra_dict.items():
                file_result[int(cid)] = {
                    "spectra_profile": d.get("spectra_profile"),
                }

            label_map = ps.labels.reshape(ps.height, ps.width).astype(np.int16, copy=False)

            results[file_key] = {
                "clusters": file_result,
                "_label_map": label_map,
                "_nav_img": (
                    nav_img_for_extraction.data
                    if hasattr(nav_img_for_extraction, "data")
                    else nav_img_for_extraction
                ),
                "_shape": (ps.height, ps.width),
            }

            if verbose:
                print(f"✅ Finished '{file_key}' - found {len(file_result)} clusters")

        except Exception as e:
            print(f"❌ Error while processing '{file_key}': {e}")
            traceback.print_exc()
            results[file_key] = {"_error": str(e)}

        finally:
            try:
                if dataset is not None and hasattr(dataset, "close"):
                    dataset.close()
            except Exception:
                pass

            try:
                if sem is not None and sem is not dataset and hasattr(sem, "close"):
                    sem.close()
            except Exception:
                pass

            dataset = None
            sem = None
            ps = None
            latent = None
            data = None
            spectra_for_extraction = None
            nav_img_for_extraction = None
            label_map = None
            umap = None
            ex = None

            gc.collect()

    return results
    
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import NMF


def nmf_on_batch_cluster_spectra(
    batch_results,
    n_components="All",
    normalised=True,
    method_args=None,
    reference_energy="auto",
    fill_value=0.0,
    energy_range=(0.0, 10.0),   # crop here
    peak_dict=None,
    peak_list=None,
):
    """
    Run NMF on all cluster-averaged spectra from batch results.

    Crops spectra to energy_range before fitting.
    """
    method_args = {} if method_args is None else dict(method_args)

    spectra_list = []

    for file_key, rec in batch_results.items():
        if not isinstance(rec, dict):
            continue
        if "_error" in rec:
            continue

        if "clusters" in rec and isinstance(rec["clusters"], dict):
            cluster_container = rec["clusters"]
        else:
            cluster_container = rec

        for cid, cluster_rec in cluster_container.items():
            if not isinstance(cluster_rec, dict):
                continue

            df = cluster_rec.get("spectra_profile", None)
            if df is None:
                continue

            try:
                e = np.asarray(df["energy"], dtype=float)
                y = np.asarray(df["intensity"], dtype=float)
            except Exception:
                continue

            if e.size == 0 or y.size == 0:
                continue

            spectra_list.append((file_key, int(cid), e, y))

    if len(spectra_list) == 0:
        raise ValueError("No valid spectra_profile entries found in batch_results.")

    # Reference energy axis
    if isinstance(reference_energy, str) and reference_energy == "auto":
        ref_energy = spectra_list[0][2]
    else:
        ref_energy = np.asarray(reference_energy, dtype=float)
        if ref_energy.ndim != 1:
            raise ValueError("reference_energy must be 1D or 'auto'.")

    # Crop reference axis to the requested range
    e0, e1 = energy_range
    ref_mask = (ref_energy >= e0) & (ref_energy <= e1)
    ref_energy = ref_energy[ref_mask]

    aligned_spectra = []
    meta_rows = []

    for file_key, cid, e, y in spectra_list:
        y_interp = np.interp(ref_energy, e, y, left=fill_value, right=fill_value)

        if normalised and y_interp.max() > 0:
            y_interp = y_interp / y_interp.max()

        aligned_spectra.append(y_interp)
        meta_rows.append({"file": file_key, "cluster_id": cid})

    X = np.vstack(aligned_spectra)

    if n_components == "All":
        n_components = min(X.shape[0], X.shape[1])
    n_components = int(n_components)

    model = NMF(n_components=n_components, **method_args)
    W = model.fit_transform(X)
    H = model.components_

    weights_df = pd.DataFrame(
        W.round(3),
        columns=[f"w_{i}" for i in range(n_components)],
    )
    metadata_df = pd.DataFrame(meta_rows)
    weights_df = pd.concat([metadata_df, weights_df], axis=1)

    components_df = pd.DataFrame(
        H.T.round(3),
        index=ref_energy,
        columns=[f"cpnt_{i}" for i in range(n_components)],
    )
    components_df.index.name = "energy"

    out = {
        "weights": weights_df,
        "components": components_df,
        "model": model,
        "metadata": metadata_df,
        "energy": ref_energy,
        "reconstruction_error": model.reconstruction_err_,
        "energy_range": energy_range,
    }

    if peak_dict is not None:
        out["peak_dict"] = peak_dict
    if peak_list is not None:
        out["peak_list"] = peak_list

    return out
    
def show_batch_unmixed_weights_and_components(
    nmf_result,
    batch_results=None,
    peak_dict=None,
    peak_list=None,
    peak_rel_threshold=0.03,   # auto peak filtering based on intensity
    peak_window_keV=0.08,      # local window around expected line energy
    save_fig=None,
):
    """
    Visualise batch NMF results with automatic peak filtering.

    Peaks are only annotated if the local intensity near the expected line
    exceeds peak_rel_threshold * component_max.
    """
    if "weights" not in nmf_result or "components" not in nmf_result:
        raise ValueError("nmf_result must contain 'weights' and 'components'.")

    weights = nmf_result["weights"].copy()
    components = nmf_result["components"].copy()

    if peak_dict is None:
        peak_dict = nmf_result.get("peak_dict", None)
        
        if peak_dict is None:
            peak_dict = {"Li_Ka": 0.055,
                        "Be_Ka": 0.108,
                        "B_Ka": 0.183,
                        "C_Ka": 0.277,
                        "N_Ka": 0.392,
                        "O_Ka": 0.525,
                        "F_Ka": 0.677,
                        "Fe_La":0.705,
                        "Ne_Ka": 0.849,
                        "Na_Ka": 1.041,
                        "Mg_Ka": 1.254,
                        "Al_Ka": 1.486,
                        "Si_Ka": 1.740,
                        "P_Ka": 2.013,
                        "S_Ka": 2.307,
                        "Cl_Ka": 2.622,
                        "Ar_Ka": 2.957,
                        "K_Ka": 3.312,
                        "Ca_Ka": 3.690,
                        "Sc_Ka": 4.090,
                        "Ti_Ka": 4.511,
                        "V_Ka": 4.952,
                        "Cr_Ka": 5.414,
                        "Mn_Ka": 5.899,
                        "Fe_Ka": 6.404,
                        "Fe_Kb":7.0580,
                        "Co_Ka": 6.930,
                        "Ni_Ka": 7.478,
                        "Cu_Ka": 8.048,
                        "Zn_Ka": 8.638,
                        "Ga_Ka": 9.251,
                        "Ge_Ka": 9.886,
                    }
    if peak_list is None and peak_dict is not None:
        peak_list = list(peak_dict.keys())

    # weight columns
    weight_cols = [c for c in weights.columns if str(c).startswith("w_")]
    if len(weight_cols) == 0:
        raise ValueError("No NMF weight columns found. Expected columns like w_0, w_1, ...")

    # sample labels
    if "file" in weights.columns and "cluster_id" in weights.columns:
        sample_labels = [
            f"{os.path.basename(str(f))} | cluster {cid}"
            for f, cid in zip(weights["file"], weights["cluster_id"])
        ]
    else:
        sample_labels = [f"sample_{i}" for i in range(len(weights))]

    weights = weights.copy()
    weights["sample_label"] = sample_labels
    weights = weights.set_index("sample_label")

    component_cols = list(components.columns)
    if len(component_cols) == 0:
        raise ValueError("No NMF component columns found in components DataFrame.")

    # energy axis
    if components.index.name is not None and "energy" in str(components.index.name).lower():
        energy_axis = components.index.to_numpy(dtype=float)
    else:
        try:
            energy_axis = np.asarray(nmf_result.get("energy", components.index), dtype=float)
        except Exception:
            energy_axis = np.asarray(components.index, dtype=float)

    def annotate_peaks(ax, y, energy_axis, peak_dict, peak_list=None):
        if peak_dict is None:
            return

        y = np.asarray(y, dtype=float)
        energy_axis = np.asarray(energy_axis, dtype=float)

        if y.size == 0 or energy_axis.size == 0:
            return

        use_peaks = peak_list if peak_list is not None else list(peak_dict.keys())

        ymax = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 1.0
        if ymax <= 0:
            return

        e_min = float(np.nanmin(energy_axis))
        e_max = float(np.nanmax(energy_axis))

        for el in use_peaks:
            if el not in peak_dict:
                continue

            peak_energy = float(peak_dict[el])
            if peak_energy < e_min or peak_energy > e_max:
                continue

            # local intensity around expected peak
            local_mask = (energy_axis >= peak_energy - peak_window_keV) & (energy_axis <= peak_energy + peak_window_keV)
            if not np.any(local_mask):
                continue

            local_peak = float(np.nanmax(y[local_mask]))
            if local_peak < (peak_rel_threshold * ymax):
                continue

            ax.axvline(
                peak_energy,
                color="grey",
                linestyle="dashed",
                linewidth=0.7,
                alpha=0.7,
            )
            ax.text(
                peak_energy - 0.08,
                ymax * 1.03,
                el,
                rotation="vertical",
                fontsize=8,
                color="black",
            )

    # widgets
    multi_select_sample = widgets.SelectMultiple(
        options=list(weights.index),
        description="Samples",
        layout=widgets.Layout(width="45%")
    )
    sample_output = widgets.Output()
    sample_table_output = widgets.Output()

    component_dropdown = widgets.Dropdown(
        options=component_cols,
        description="Component",
        layout=widgets.Layout(width="45%")
    )
    component_output = widgets.Output()
    component_table_output = widgets.Output()

    with sample_table_output:
        display(weights[weight_cols + [c for c in ["file", "cluster_id"] if c in weights.columns]].round(3))

    def on_sample_change(change):
        sample_output.clear_output()
        with sample_output:
            selected = list(change.new)
            if len(selected) == 0:
                print("Select one or more samples.")
                return

            display(weights.loc[selected, weight_cols + [c for c in ["file", "cluster_id"] if c in weights.columns]].round(3))

            for sample in selected:
                fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=110)
                vals = weights.loc[sample, weight_cols].to_numpy(dtype=float)

                ax.bar(np.arange(len(weight_cols)), vals, width=0.6)
                ax.set_xticks(np.arange(len(weight_cols)))
                ax.set_xticklabels([c.replace("w_", "") for c in weight_cols])
                ax.set_ylabel("Abundance coefficient")
                ax.set_xlabel("NMF component ID")
                ax.set_title(str(sample))
                plt.tight_layout()
                plt.show()

                if save_fig is not None:
                    save_fig(fig)

    multi_select_sample.observe(on_sample_change, names="value")

    with component_table_output:
        display(components.round(3))

    def on_component_change(change):
        component_output.clear_output()
        with component_output:
            cpnt = change.new
            y = components[cpnt].to_numpy(dtype=float)

            fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), dpi=110)
            ax.plot(energy_axis, y, linewidth=1.2)

            annotate_peaks(ax, y, energy_axis, peak_dict, peak_list)

            ax.set_xlim(0, 10)
            ax.set_xlabel("Energy / keV")
            ax.set_ylabel("Intensity")
            ax.set_title(cpnt)
            plt.tight_layout()
            plt.show()

            if save_fig is not None:
                save_fig(fig)

    component_dropdown.observe(on_component_change, names="value")

    sample_box = widgets.VBox([
        widgets.HBox([multi_select_sample]),
        sample_table_output,
        sample_output
    ])

    component_box = widgets.VBox([
        widgets.HBox([component_dropdown]),
        component_table_output,
        component_output
    ])

    tabs = widgets.Tab(children=[sample_box, component_box])
    tabs.set_title(0, "Weights")
    tabs.set_title(1, "Components")

    display(tabs)
    
import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def show_batch_abundance_map(
    batch_results,
    nmf_result,
    save_fig=None,
):
    """
    Show NMF abundance maps for batch-processing output.

    Expected structure
    ------------------
    batch_results[file_key] = {
        "clusters": {...},      # not used directly here
        "_label_map": np.ndarray of shape (H, W),
        "_nav_img": np.ndarray or None,
        "_shape": (H, W)
    }

    nmf_result["weights"] must contain:
        - file
        - cluster_id
        - w_0, w_1, ...
    """

    if "weights" not in nmf_result:
        raise ValueError("nmf_result must contain a 'weights' DataFrame.")

    weights = nmf_result["weights"].copy()
    weight_cols = [c for c in weights.columns if str(c).startswith("w_")]
    if len(weight_cols) == 0:
        raise ValueError("No NMF weight columns found (expected w_0, w_1, ...).")

    if "file" not in weights.columns or "cluster_id" not in weights.columns:
        raise ValueError("weights must contain 'file' and 'cluster_id' columns.")
        
    
    # ------------------------------------------------------------------
    # Global per-component normalisation factors (dataset-wide)
    # ------------------------------------------------------------------
    global_max = {}

    for col in weight_cols:
        vmax = weights[col].max()
        if vmax <= 0 or not np.isfinite(vmax):
            global_max[col] = 1.0  # avoid division-by-zero
        else:
            global_max[col] = float(vmax)


    # Files that actually have label maps
    file_keys = [
        fk for fk, rec in batch_results.items()
        if isinstance(rec, dict) and "_label_map" in rec
    ]
    if len(file_keys) == 0:
        raise ValueError("No '_label_map' entries found in batch_results.")

    def build_rgb_map(file_key, phases):
        rec = batch_results[file_key]
        label_map = np.asarray(rec["_label_map"])
        nav_img = rec.get("_nav_img", None)

        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=float)

        # weights for this file only
        file_weights = weights[weights["file"] == file_key].copy()
        if file_weights.empty:
            raise ValueError(f"No NMF weights found for file '{file_key}'.")

        file_weights = file_weights.set_index("cluster_id")

        # make one channel per selected component
        for channel_idx, phase in enumerate(phases):
            if phase == "None":
                continue
            if phase not in weight_cols:
                continue

            channel_img = np.zeros((h, w), dtype=float)

            for cluster_id in np.unique(label_map):
                if cluster_id < 0:
                    continue
                if cluster_id not in file_weights.index:
                    continue

                value = float(file_weights.loc[cluster_id, phase])
                channel_img[label_map == cluster_id] = value

            # normalise channel for display
            channel_img = channel_img / global_max[phase]
            channel_img = np.clip(channel_img, 0.0, 1.0)

            rgb[:, :, channel_idx] = channel_img

        fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=110)

        axs[0].imshow(rgb, interpolation="none")
        axs[0].set_title(f"RGB abundance map\n{os.path.basename(str(file_key))}")
        axs[0].axis("off")

        if nav_img is not None:
            axs[1].imshow(nav_img, cmap="gray", interpolation="none")
            axs[1].set_title("Navigation image")
            axs[1].axis("off")
        else:
            axs[1].text(0.5, 0.5, "No navigation image stored", ha="center", va="center")
            axs[1].axis("off")

        plt.tight_layout()
        plt.show()

        if save_fig is not None:
            save_fig(fig)

        return fig

    # Dropdown options
    component_options = [("None", "None")] + [(c, c) for c in weight_cols]

    dropdown_file = widgets.Dropdown(
        options=[(os.path.basename(str(k)), k) for k in file_keys],
        value=file_keys[0],
        description="File:"
    )
    dropdown_r = widgets.Dropdown(options=component_options, value="None", description="Red:")
    dropdown_g = widgets.Dropdown(options=component_options, value="None", description="Green:")
    dropdown_b = widgets.Dropdown(options=component_options, value="None", description="Blue:")

    plots_output = widgets.Output()

    def redraw(*args):
        plots_output.clear_output()
        with plots_output:
            build_rgb_map(
                dropdown_file.value,
                [dropdown_r.value, dropdown_g.value, dropdown_b.value]
            )

    dropdown_file.observe(lambda change: redraw(), names="value")
    dropdown_r.observe(lambda change: redraw(), names="value")
    dropdown_g.observe(lambda change: redraw(), names="value")
    dropdown_b.observe(lambda change: redraw(), names="value")

    display(
        widgets.VBox([
            dropdown_file,
            widgets.HBox([dropdown_r, dropdown_g, dropdown_b]),
            plots_output
        ])
    )

    redraw()
    
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def show_batch_abundance_map_interactive_tight(
    batch_results,
    nmf_result,
    grid_shape=None,
    dpi=140,
    save_fig=None,
):
    if "weights" not in nmf_result:
        raise ValueError("nmf_result must contain a 'weights' DataFrame.")

    weights = nmf_result["weights"].copy()
    weight_cols = [c for c in weights.columns if str(c).startswith("w_")]
    if len(weight_cols) == 0:
        raise ValueError("No NMF weight columns found (expected w_0, w_1, ...).")

    if "file" not in weights.columns or "cluster_id" not in weights.columns:
        raise ValueError("weights must contain 'file' and 'cluster_id' columns.")

    def site_sort_key(path_like):
        base = os.path.basename(str(path_like))
        site_match = re.search(r"(?i)\bsite\s*(\d+)\b", base)
        site_num = int(site_match.group(1)) if site_match else 10**9
        nums = [int(x) for x in re.findall(r"\d+", base)]
        return (site_num, nums, base.lower())

    file_keys = [
        fk for fk, rec in batch_results.items()
        if isinstance(rec, dict) and "_label_map" in rec
    ]
    if len(file_keys) == 0:
        raise ValueError("No files with '_label_map' found in batch_results.")

    file_keys = sorted(file_keys, key=site_sort_key)

    if grid_shape is None:
        n_rows = 1
        n_cols = len(file_keys)
    else:
        n_rows, n_cols = grid_shape
        if n_rows * n_cols < len(file_keys):
            raise ValueError(
                f"grid_shape={grid_shape} is too small for {len(file_keys)} files."
            )

    # ------------------------------------------------------------------
    # Pre-compute global max per component across ALL files
    # ------------------------------------------------------------------
    global_max = {}
    for col in weight_cols:
        vmax = weights[col].max()
        global_max[col] = float(vmax) if (np.isfinite(vmax) and vmax > 0) else 1.0

    component_options = [("None", "None")] + [(c, c) for c in weight_cols]
    dropdown_r = widgets.Dropdown(options=component_options, value="None", description="Red:")
    dropdown_g = widgets.Dropdown(options=component_options, value="None", description="Green:")
    dropdown_b = widgets.Dropdown(options=component_options, value="None", description="Blue:")

    out = widgets.Output()

    def build_rgb_for_file(file_key, phases):
        rec = batch_results[file_key]
        label_map = np.asarray(rec["_label_map"])
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=float)

        file_weights = weights[weights["file"] == file_key].copy()
        if file_weights.empty:
            raise ValueError(f"No NMF weights found for file '{file_key}'.")

        file_weights = file_weights.set_index("cluster_id")

        for channel_idx, phase in enumerate(phases[:3]):
            if phase == "None":
                continue
            if phase not in weight_cols:
                continue

            channel_img = np.zeros((h, w), dtype=float)

            for cluster_id in np.unique(label_map):
                if cluster_id < 0:
                    continue
                if cluster_id not in file_weights.index:
                    continue

                value = float(file_weights.loc[cluster_id, phase])
                channel_img[label_map == cluster_id] = value

            # Normalise by global max so colours are comparable across all files
            channel_img = channel_img / global_max[phase]
            channel_img = np.clip(channel_img, 0.0, 1.0)

            rgb[:, :, channel_idx] = channel_img

        return rgb

    def build_nav_for_file(file_key):
        rec = batch_results[file_key]
        nav_img = rec.get("_nav_img", None)
        if nav_img is None:
            label_map = np.asarray(rec["_label_map"])
            return np.zeros_like(label_map, dtype=float)
        return np.asarray(nav_img)

    def make_stitched_canvas(images, n_rows, n_cols, fill_value=0.0):
        if len(images) == 0:
            raise ValueError("No images to stitch.")

        first = np.asarray(images[0])
        if first.ndim == 2:
            h, w = first.shape
            canvas = np.full((n_rows * h, n_cols * w), fill_value, dtype=first.dtype)
        elif first.ndim == 3:
            h, w, ch = first.shape
            canvas = np.full((n_rows * h, n_cols * w, ch), fill_value, dtype=first.dtype)
        else:
            raise ValueError(f"Unsupported image ndim={first.ndim}")

        for idx, img in enumerate(images):
            r = idx // n_cols
            c = idx % n_cols
            if r >= n_rows:
                break

            img = np.asarray(img)
            if img.shape[:2] != (h, w):
                raise ValueError(
                    f"All images must have the same shape. Expected {(h, w)}, got {img.shape[:2]}."
                )

            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

        return canvas

    def render():
            phases = [dropdown_r.value, dropdown_g.value, dropdown_b.value]

            rgb_images = [build_rgb_for_file(fk, phases) for fk in file_keys]
            nav_images = [build_nav_for_file(fk) for fk in file_keys]

            rgb_canvas = make_stitched_canvas(rgb_images, n_rows, n_cols, fill_value=0.0)
            nav_canvas = make_stitched_canvas(nav_images, n_rows, n_cols, fill_value=0.0)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=dpi)

            ax1.imshow(nav_canvas, cmap="gray", interpolation="nearest")
            ax1.axis("off")
            ax1.set_title("Navigation image")

            ax2.imshow(rgb_canvas, interpolation="nearest")
            ax2.axis("off")
            ax2.set_title("RGB abundance map")

            plt.tight_layout(pad=0.5)
            plt.show()

            if save_fig is not None:
                save_fig(fig)

    def on_change(change):
        out.clear_output(wait=True)
        with out:
            render()

    dropdown_r.observe(on_change, names="value")
    dropdown_g.observe(on_change, names="value")
    dropdown_b.observe(on_change, names="value")

    display(widgets.VBox([widgets.HBox([dropdown_r, dropdown_g, dropdown_b]), out]))

    render()
    
    
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.decomposition import NMF

from typing import Iterable, Dict, Any
import os
import gc
import traceback
import numpy as np

from sigma.utils import normalisation as norm 
from sigma.utils import visualisation as visual
from sigma.utils.load import SEMDataset
from sigma.src.utils import same_seeds
from sigma.src.dim_reduction import Experiment
from sigma.models.autoencoder import AutoEncoder
from sigma.src.segmentation import PixelSegmenter
from sigma.gui import gui
from sigma.utils.loadtem import TEMDataset

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def show_batch_abundance_map_interactive_tight_overlap(
    batch_results,
    nmf_result,
    grid_shape=None,
    dpi=140,
    save_fig=None,
    overlap_min_frac=0.03,
    overlap_max_frac=0.10,
    overlap_clamp_frac=0.25,
    overlap_score_floor=-0.05,
    overlap_downsample=2,
    verbose=True,
    force_vertical_overlap_zero=False,   # <-- add this
):
    from collections import Counter

    if "weights" not in nmf_result:
        raise ValueError("nmf_result must contain a 'weights' DataFrame.")

    weights = nmf_result["weights"].copy()
    weight_cols = [c for c in weights.columns if str(c).startswith("w_")]
    if len(weight_cols) == 0:
        raise ValueError("No NMF weight columns found (expected w_0, w_1, ...).")

    if "file" not in weights.columns or "cluster_id" not in weights.columns:
        raise ValueError("weights must contain 'file' and 'cluster_id' columns.")

    def site_sort_key(path_like):
        base = os.path.basename(str(path_like))
        site_match = re.search(r"(?i)\bsite\s*(\d+)\b", base)
        site_num = int(site_match.group(1)) if site_match else 10**9
        nums = [int(x) for x in re.findall(r"\d+", base)]
        return (site_num, nums, base.lower())

    file_keys = [
        fk for fk, rec in batch_results.items()
        if isinstance(rec, dict) and "_label_map" in rec
    ]
    if len(file_keys) == 0:
        raise ValueError("No files with '_label_map' found in batch_results.")

    file_keys = sorted(file_keys, key=site_sort_key)

    if grid_shape is None:
        n_rows = 1
        n_cols = len(file_keys)
    else:
        n_rows, n_cols = grid_shape
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("grid_shape must contain positive integers (n_rows, n_cols).")
        if n_rows * n_cols < len(file_keys):
            raise ValueError(
                f"grid_shape={grid_shape} is too small for {len(file_keys)} files."
            )

    # ------------------------------------------------------------------
    # Global per-component normalisation
    # ------------------------------------------------------------------
    global_max = {}
    for col in weight_cols:
        vmax = weights[col].max()
        global_max[col] = float(vmax) if (np.isfinite(vmax) and vmax > 0) else 1.0

    # ------------------------------------------------------------------
    # Overlap estimation helpers
    # ------------------------------------------------------------------
    def _to_gray(img):
        img = np.asarray(img)
        if img.ndim == 2:
            return img.astype(np.float32, copy=False)
        if img.ndim == 3 and img.shape[2] >= 3:
            return (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.float32)
        return img.mean(axis=-1).astype(np.float32)

    def _ncc(a, b, eps=1e-8):
        a = a.astype(np.float32, copy=False) - float(np.mean(a))
        b = b.astype(np.float32, copy=False) - float(np.mean(b))
        sa, sb = float(np.std(a)), float(np.std(b))
        if sa < eps or sb < eps:
            return -np.inf
        return float(np.mean((a / (sa + eps)) * (b / (sb + eps))))

    def estimate_overlap_ncc(img_a, img_b, axis="x", min_frac=0.03, max_frac=0.10, downsample=2):
        a = _to_gray(img_a)
        b = _to_gray(img_b)
        if a.shape != b.shape:
            raise ValueError(f"Tile shapes differ: A={a.shape}, B={b.shape}")

        h, w = a.shape
        ds = max(1, int(downsample))

        if axis == "x":
            min_px = max(1, int(w * min_frac))
            max_px = max(min_px, int(w * max_frac))
        else:
            min_px = max(1, int(h * min_frac))
            max_px = max(min_px, int(h * max_frac))

        best_ov, best_score = min_px, -np.inf

        if ds > 1:
            a, b = a[::ds, ::ds], b[::ds, ::ds]
            h2, w2 = a.shape
            min_px2 = max(1, int(min_px / ds))
            max_px2 = max(min_px2, int(max_px / ds))
            if axis == "x":
                for ov2 in range(min_px2, max_px2 + 1):
                    score = _ncc(a[:, w2 - ov2:], b[:, :ov2])
                    if score > best_score:
                        best_score, best_ov = score, ov2 * ds
            else:
                for ov2 in range(min_px2, max_px2 + 1):
                    score = _ncc(a[h2 - ov2:, :], b[:ov2, :])
                    if score > best_score:
                        best_score, best_ov = score, ov2 * ds
        else:
            if axis == "x":
                for ov in range(min_px, max_px + 1):
                    score = _ncc(a[:, w - ov:], b[:, :ov])
                    if score > best_score:
                        best_score, best_ov = score, ov
            else:
                for ov in range(min_px, max_px + 1):
                    score = _ncc(a[h - ov:, :], b[:ov, :])
                    if score > best_score:
                        best_score, best_ov = score, ov

        return int(best_ov), float(best_score)

    def as_tile_grid(items_1d, n_rows, n_cols):
        tiles, idx = [], 0
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                row.append(items_1d[idx] if idx < len(items_1d) else None)
                idx += 1
            tiles.append(row)
        return tiles

    def print_horizontal_overlaps(nav_tiles_local, file_keys, n_rows, n_cols,
                                  ov_x, row_med_x, med_x):
        """
        Print horizontal overlap for each adjacent pair (r,c) -> (r,c+1).
        Reports measured ov_x where available; otherwise indicates fallback used.
        """
    
        # find a valid tile width
        w0 = None
        for row in nav_tiles_local:
            for t in row:
                if t is not None:
                    w0 = int(np.asarray(t).shape[1])
                    break
            if w0 is not None:
                break
    
        if w0 is None:
            print("[overlap] Could not determine tile width (no tiles found).")
            return
    
        # build a parallel grid of file basenames for nice printing
        names = []
        idx = 0
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                if idx < len(file_keys):
                    row.append(os.path.basename(str(file_keys[idx])))
                else:
                    row.append(None)
                idx += 1
            names.append(row)
    
        print("\n================ HORIZONTAL OVERLAPS (per row) ================")
        print(f"Tile width ≈ {w0} px")
        if np.isfinite(med_x):
            print(f"Global median overlap ≈ {float(med_x):.1f} px ({100*float(med_x)/w0:.2f}%)")
        else:
            print("Global median overlap: n/a")
    
        for r in range(n_rows):
            # check if row has at least two tiles
            row_has_any = any(t is not None for t in nav_tiles_local[r])
            if not row_has_any:
                continue
    
            print(f"\nRow {r}:")
            for c in range(n_cols - 1):
                left_name = names[r][c]
                right_name = names[r][c + 1]
    
                left_tile = nav_tiles_local[r][c]
                right_tile = nav_tiles_local[r][c + 1]
    
                # if either missing, skip
                if left_tile is None or right_tile is None:
                    continue
    
                # prefer measured overlap
                source = "measured"
                ov = np.nan
    
                if ov_x is not None and r < ov_x.shape[0] and c < ov_x.shape[1] and np.isfinite(ov_x[r, c]):
                    ov = float(ov_x[r, c])
                else:
                    # fallback hierarchy matches your stitcher
                    if row_med_x is not None and r < len(row_med_x) and np.isfinite(row_med_x[r]):
                        ov = float(row_med_x[r])
                        source = "row_median_fallback"
                    elif np.isfinite(med_x):
                        ov = float(med_x)
                        source = "global_median_fallback"
                    else:
                        ov = 0.0
                        source = "zero_fallback"
    
                ov_i = int(max(0, round(ov)))
                ov_pct = 100.0 * ov_i / w0
    
                print(
                    f"  (r{r}, c{c}) '{left_name}'  ->  (r{r}, c{c+1}) '{right_name}'"
                    f"   overlap = {ov_i:4d}px  ({ov_pct:5.2f}%)   [{source}]"
                )
    
        print("\n==============================================================\n")
    

    def compute_grid_overlaps_from_nav(nav_tiles, min_frac, max_frac, clamp_frac, score_floor, downsample):
        n_rows_local = len(nav_tiles)
        n_cols_local = max((len(r) for r in nav_tiles), default=0)

        ov_x = np.full((n_rows_local, max(0, n_cols_local - 1)), np.nan, dtype=float)
        ov_y = np.full((max(0, n_rows_local - 1), n_cols_local), np.nan, dtype=float)

        for r in range(n_rows_local):
            for c in range(n_cols_local - 1):
                a = nav_tiles[r][c] if c < len(nav_tiles[r]) else None
                b = nav_tiles[r][c + 1] if (c + 1) < len(nav_tiles[r]) else None
                if a is None or b is None:
                    continue
                try:
                    ov, score = estimate_overlap_ncc(a, b, axis="x", min_frac=min_frac,
                                                     max_frac=max_frac, downsample=downsample)
                    if score >= score_floor:
                        ov_x[r, c] = ov
                except Exception:
                    continue

        for r in range(n_rows_local - 1):
            for c in range(n_cols_local):
                a = nav_tiles[r][c] if c < len(nav_tiles[r]) else None
                b = nav_tiles[r + 1][c] if c < len(nav_tiles[r + 1]) else None
                if a is None or b is None:
                    continue
                try:
                    ov, score = estimate_overlap_ncc(a, b, axis="y", min_frac=min_frac,
                                                     max_frac=max_frac, downsample=downsample)
                    if score >= score_floor:
                        ov_y[r, c] = ov
                except Exception:
                    continue

        med_x = np.nanmedian(ov_x) if np.any(np.isfinite(ov_x)) else np.nan
        med_y = np.nanmedian(ov_y) if np.any(np.isfinite(ov_y)) else np.nan

        def _clamp(arr, med):
            if not np.isfinite(med):
                return arr
            lo, hi = med * (1 - clamp_frac), med * (1 + clamp_frac)
            out = arr.copy()
            m = np.isfinite(out)
            out[m] = np.clip(out[m], lo, hi)
            return out

        ov_x = _clamp(ov_x, med_x)
        ov_y = _clamp(ov_y, med_y)

        row_med_x = []
        for r in range(ov_x.shape[0]):
            vals = ov_x[r, :][np.isfinite(ov_x[r, :])]
            row_med_x.append(float(np.median(vals)) if vals.size else np.nan)

        boundary_med_y = []
        for r in range(ov_y.shape[0]):
            vals = ov_y[r, :][np.isfinite(ov_y[r, :])]
            boundary_med_y.append(float(np.median(vals)) if vals.size else np.nan)

        return ov_x, ov_y, med_x, med_y, row_med_x, boundary_med_y

    def _conform_width(rm, target_w, fill_value, is_rgb):
        h, w = rm.shape[:2]
        if w == target_w:
            return rm
        if w > target_w:
            return rm[:, :target_w, :] if is_rgb else rm[:, :target_w]
        pad_w = target_w - w
        pad = np.full((h, pad_w, rm.shape[2]), fill_value, dtype=rm.dtype) if is_rgb \
              else np.full((h, pad_w), fill_value, dtype=rm.dtype)
        return np.concatenate([rm, pad], axis=1)

    def stitch_grid_with_overlaps(tiles, ov_x, ov_y, med_x=np.nan, med_y=np.nan,
                                  row_med_x=None, boundary_med_y=None, fill_value=0.0):
        first = None
        for row in tiles:
            for t in row:
                if t is not None:
                    first = np.asarray(t)
                    break
            if first is not None:
                break
        if first is None:
            raise ValueError("No tiles to stitch.")

        is_rgb = first.ndim == 3

        # Horizontal stitch each row
        row_mosaics = []
        for r, row in enumerate(tiles):
            existing = [(c, np.asarray(t)) for c, t in enumerate(row) if t is not None]
            if not existing:
                row_mosaics.append(None)
                continue

            _, mosaic = existing[0]
            for c, tile in existing[1:]:
                ov = np.nan
                if (ov_x is not None and r < ov_x.shape[0] and
                        (c - 1) >= 0 and (c - 1) < ov_x.shape[1] and
                        np.isfinite(ov_x[r, c - 1])):
                    ov = ov_x[r, c - 1]

                if not np.isfinite(ov):
                    if row_med_x is not None and r < len(row_med_x) and np.isfinite(row_med_x[r]):
                        ov = row_med_x[r]
                    elif np.isfinite(med_x):
                        ov = med_x
                    else:
                        ov = 0.0

                ov = int(max(0, round(float(ov))))
                if ov > 0:
                    mosaic = mosaic[:, :-ov, :] if is_rgb else mosaic[:, :-ov]
                mosaic = np.concatenate([mosaic, tile], axis=1)

            row_mosaics.append(mosaic)

        # Conform all row widths before vertical stitch
        valid_widths = [rm.shape[1] for rm in row_mosaics if rm is not None]
        if not valid_widths:
            raise ValueError("No row mosaics to stitch.")

        target_w = Counter(valid_widths).most_common(1)[0][0]
        row_mosaics = [
            _conform_width(rm, target_w, fill_value, is_rgb) if rm is not None else None
            for rm in row_mosaics
        ]

        # Vertical stitch
        stitched, stitched_r = None, None
        for r, rm in enumerate(row_mosaics):
            if rm is not None:
                stitched, stitched_r = rm, r
                break

        for r in range(stitched_r + 1, len(row_mosaics)):
            rm = row_mosaics[r]
            if rm is None:
                continue

            ov = np.nan
            boundary = r - 1
            if (boundary_med_y is not None and boundary < len(boundary_med_y) and
                    np.isfinite(boundary_med_y[boundary])):
                ov = boundary_med_y[boundary]
            elif np.isfinite(med_y):
                ov = med_y
            else:
                ov = 0.0

            ov = int(max(0, round(float(ov))))
            if ov > 0:
                stitched = stitched[:-ov, :, :] if is_rgb else stitched[:-ov, :]
            stitched = np.concatenate([stitched, rm], axis=0)

        return stitched

    # ------------------------------------------------------------------
    # Nav + overlap computation (runs once at startup)
    # ------------------------------------------------------------------
    def build_nav_for_file(file_key):
        rec = batch_results[file_key]
        nav_img = rec.get("_nav_img", None)
        if nav_img is None:
            label_map = np.asarray(rec["_label_map"])
            return np.zeros_like(label_map, dtype=float)
        return np.asarray(nav_img)

    nav_images_all = [build_nav_for_file(fk) for fk in file_keys]
    nav_tiles = as_tile_grid(nav_images_all, n_rows, n_cols)

    ov_x, ov_y, med_x, med_y, row_med_x, boundary_med_y = compute_grid_overlaps_from_nav(
        nav_tiles,
        min_frac=overlap_min_frac,
        max_frac=overlap_max_frac,
        clamp_frac=overlap_clamp_frac,
        score_floor=overlap_score_floor,
        downsample=overlap_downsample,
    )

    if force_vertical_overlap_zero:
        if verbose:
            print("[overlap] Forcing vertical overlap to zero (no vertical cropping).")
        if ov_y is not None:
            ov_y[:] = 0.0
        med_y = 0.0
        if boundary_med_y is not None:
            boundary_med_y = [0.0 for _ in boundary_med_y]

    if verbose:
        mx = f"{float(med_x):.1f}px" if np.isfinite(med_x) else "n/a"
        my = f"{float(med_y):.1f}px" if np.isfinite(med_y) else "n/a"
        print(f"[overlap] median horizontal ≈ {mx}, median vertical ≈ {my}")
        hx = int(np.sum(np.isfinite(ov_x))) if ov_x.size else 0
        hy = int(np.sum(np.isfinite(ov_y))) if ov_y.size else 0
        print(f"[overlap] valid estimates: horizontal={hx}, vertical={hy}")

    # ------------------------------------------------------------------
    # RGB builder
    # ------------------------------------------------------------------
    def build_rgb_for_file(file_key, phases):
        rec = batch_results[file_key]
        label_map = np.asarray(rec["_label_map"])
        h, w = label_map.shape
        rgb = np.zeros((h, w, 3), dtype=float)

        file_weights = weights[weights["file"] == file_key].copy()
        if file_weights.empty:
            raise ValueError(f"No NMF weights found for file '{file_key}'.")
        file_weights = file_weights.set_index("cluster_id")

        for channel_idx, phase in enumerate(phases[:3]):
            if phase == "None" or phase not in weight_cols:
                continue
            channel_img = np.zeros((h, w), dtype=float)
            for cluster_id in np.unique(label_map):
                if cluster_id < 0 or cluster_id not in file_weights.index:
                    continue
                channel_img[label_map == cluster_id] = float(file_weights.loc[cluster_id, phase])
            channel_img = np.clip(channel_img / global_max[phase], 0.0, 1.0)
            rgb[:, :, channel_idx] = channel_img

        return rgb

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------
    component_options = [("None", "None")] + [(c, c) for c in weight_cols]
    dropdown_r = widgets.Dropdown(options=component_options, value="None", description="Red:")
    dropdown_g = widgets.Dropdown(options=component_options, value="None", description="Green:")
    dropdown_b = widgets.Dropdown(options=component_options, value="None", description="Blue:")
    out = widgets.Output()

    def render():
        phases = [dropdown_r.value, dropdown_g.value, dropdown_b.value]

        rgb_images = [build_rgb_for_file(fk, phases) for fk in file_keys]
        nav_images_render = [build_nav_for_file(fk) for fk in file_keys]

        rgb_tiles = as_tile_grid(rgb_images, n_rows, n_cols)
        nav_tiles_local = as_tile_grid(nav_images_render, n_rows, n_cols)

        
        print_horizontal_overlaps(
            nav_tiles_local=nav_tiles_local,
            file_keys=file_keys,
            n_rows=n_rows,
            n_cols=n_cols,
            ov_x=ov_x,
            row_med_x=row_med_x,
            med_x=med_x,
        )


        rgb_canvas = stitch_grid_with_overlaps(
            rgb_tiles, ov_x, ov_y,
            med_x=med_x, med_y=med_y,
            row_med_x=row_med_x, boundary_med_y=boundary_med_y,
            fill_value=0.0,
        )
        nav_canvas = stitch_grid_with_overlaps(
            nav_tiles_local, ov_x, ov_y,
            med_x=med_x, med_y=med_y,
            row_med_x=row_med_x, boundary_med_y=boundary_med_y,
            fill_value=0.0,
        )
        # --- DEBUG: draw correct seam lines on nav mosaic (row 0) ---
        fig_debug, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=dpi)
        ax.imshow(nav_canvas, cmap="gray", interpolation="nearest")
        ax.axis("off")
        ax.set_title("Nav canvas with TRUE join (seam) lines")
        
        # find a valid tile width
        w0 = None
        for row in nav_tiles_local:
            for t in row:
                if t is not None:
                    w0 = np.asarray(t).shape[1]
                    break
            if w0 is not None:
                break
        
        if w0 is not None:
            r = 0  # first row seams
            x = 0
            # draw seam between tile c and c+1 at x += (w0 - overlap_c)
            for c in range(n_cols - 1):
                # overlap for this join
                ov = np.nan
                if ov_x is not None and r < ov_x.shape[0] and c < ov_x.shape[1] and np.isfinite(ov_x[r, c]):
                    ov = ov_x[r, c]
                elif row_med_x is not None and r < len(row_med_x) and np.isfinite(row_med_x[r]):
                    ov = row_med_x[r]
                elif np.isfinite(med_x):
                    ov = med_x
                else:
                    ov = 0.0
        
                ov = int(max(0, round(float(ov))))
                x = x + (w0 - ov)  # THIS is the join location
                ax.axvline(x, color="yellow", linewidth=0.9, alpha=0.8)
        
        plt.tight_layout()
        plt.show()

        # --- DEBUG: expected size vs actual size ---
        tile0 = next(t for t in nav_images_render if t is not None)
        h0, w0 = np.asarray(tile0).shape[:2]
        
        print("Single tile (h,w):", (h0, w0))
        print("Naive canvas (h,w):", (n_rows*h0, n_cols*w0))
        print("Actual nav_canvas:", np.asarray(nav_canvas).shape[:2])
        print("Actual rgb_canvas:", np.asarray(rgb_canvas).shape[:2])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=dpi)
        ax1.imshow(nav_canvas, cmap="gray", interpolation="nearest")
        ax1.axis("off")
        ax1.set_title("Navigation image")
        ax2.imshow(rgb_canvas, interpolation="nearest")
        ax2.axis("off")
        ax2.set_title("RGB abundance map")
        plt.tight_layout(pad=0.5)
        plt.show()

        if save_fig is not None:
            save_fig(fig)

    def on_change(change):
        out.clear_output(wait=True)
        with out:
            render()

    dropdown_r.observe(on_change, names="value")
    dropdown_g.observe(on_change, names="value")
    dropdown_b.observe(on_change, names="value")

    display(widgets.VBox([widgets.HBox([dropdown_r, dropdown_g, dropdown_b]), out]))

    with out:
        render()
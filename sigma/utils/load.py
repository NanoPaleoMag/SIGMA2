import os
import numpy as np
import hyperspy.api as hs

from typing import Union, Tuple, List, Callable
from pathlib import Path
from PIL import Image, ImageOps
from os.path import isfile, join
from skimage.transform import resize
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.signal2d import Signal2D
from .base import BaseDataset
from scipy.signal import find_peaks
import h5py

import copy


class SEMDataset(BaseDataset):
    def __init__(self, file_path: Union[str, Path], nag_file_path: Union[str, Path] = None):
        super().__init__(file_path)
        self.nav_img_feature = None

        if str(file_path).endswith(".h5oina"):
            self._load_h5oina(file_path)
        else:
            self.base_dataset = hs.load(file_path)
            


            if str(file_path).endswith('.bcf'):
                for dataset in self.base_dataset:
                    if (self.nav_img is None) and isinstance(dataset, Signal2D):
                        self.original_nav_img = dataset
                        self.nav_img = dataset
                    elif (self.nav_img is not None) and isinstance(dataset, Signal2D):
                        if sum(dataset.data.shape) < sum(self.nav_img.data.shape):
                            self.original_nav_img = dataset
                            self.nav_img = dataset
                    elif isinstance(dataset, hs.signals.EDSSEMSpectrum):
                        self.original_spectra = dataset
                        self.spectra = dataset

            elif str(file_path).endswith('.hspy'):
                if nag_file_path:
                    nav_img = hs.load(nag_file_path)
                else:
                    nav_img = Signal2D(self.base_dataset.sum(axis=2).data).T

                self.original_nav_img = nav_img
                self.nav_img = nav_img
                self.original_spectra = self.base_dataset
                self.spectra = self.base_dataset

            elif str(file_path).endswith('.rpl'):
                self.base_dataset = self.base_dataset.transpose()
                self.base_dataset.axes_manager[2].name = 'Energy'
                self.base_dataset.axes_manager[2].units = 'keV'
                self.base_dataset.axes_manager[0].name = 'width'
                self.base_dataset.axes_manager[1].name = 'height'
                self.base_dataset.set_signal_type('EDS_SEM')

                scale = offset = None
                emsa_path = str(file_path)[:-4] + '.txt'
                if isfile(emsa_path):
                    with open(emsa_path) as f:
                        for line in f:
                            if "#FORMAT" in line:
                                for line in f:
                                    if line.startswith("#XPERCHAN"):
                                        scale = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                                    elif line.startswith("#OFFSET"):
                                        offset = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                                        if "-" in line:
                                            offset = -abs(offset)
                                break
                else:
                    print("Could not find EMSA file — using default energy calibration")
                    self.base_dataset.axes_manager[2].scale = 0.01
                    self.base_dataset.axes_manager[2].offset = -0.2

                par_path = str(file_path)[:-4] + '.par'
                if isfile(par_path):
                    print(f"Reading scaling from {par_path}")
                    params = read_par(par_path)
                    self.base_dataset.axes_manager[0].scale = params['scale_x']
                    self.base_dataset.axes_manager[1].scale = params['scale_y']
                    self.base_dataset.axes_manager[0].units = 'μm'
                    self.base_dataset.axes_manager[1].units = 'μm'
                    if scale and offset:
                        self.base_dataset.axes_manager[2].scale = scale
                        self.base_dataset.axes_manager[2].offset = offset

                sum_image = self.base_dataset.sum(axis=2)
                if nag_file_path:
                    nav_img = hs.load(nag_file_path)
                    if nav_img.data.dtype.names is not None:
                        rgb = np.stack([nav_img.data[n] for n in ('R', 'G', 'B')], axis=-1)
                        nav_img.data = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
                    elif nav_img.data.ndim == 3 and nav_img.data.shape[-1] == 3:
                        nav_img.data = np.dot(nav_img.data[..., :3], [0.2989, 0.5870, 0.1140])

                    target_shape = sum_image.data.shape[-2:]
                    if nav_img.data.shape != target_shape:
                        nav_img.data = resize(nav_img.data, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)

                    nav_img.axes_manager[0].units = sum_image.axes_manager[0].units
                    nav_img.axes_manager[1].units = sum_image.axes_manager[1].units
                    nav_img.axes_manager[0].scale = (
                        sum_image.axes_manager[0].scale * sum_image.axes_manager[0].size / nav_img.axes_manager[0].size
                    )
                    nav_img.axes_manager[1].scale = (
                        sum_image.axes_manager[1].scale * sum_image.axes_manager[1].size / nav_img.axes_manager[1].size
                    )
                else:
                    nav_img = sum_image

                self.original_nav_img = nav_img
                self.nav_img = nav_img
                self.original_spectra = self.base_dataset
                self.spectra = self.base_dataset
                
            # for .bcf files:
        if file_path.endswith('.bcf'):
            for dataset in self.base_dataset:
                if (self.nav_img is None) and (type(dataset) is Signal2D):
                    self.original_nav_img = dataset
                    self.nav_img = dataset  # load BSE data
                elif (self.nav_img is not None) and (type(dataset) is Signal2D):
                    old_w, old_h = self.nav_img.data.shape
                    new_w, new_h = dataset.data.shape
                    if (new_w + new_h) < (old_w + old_h):
                        self.original_nav_img = dataset
                        self.nav_img = dataset
                elif type(dataset) is EDSSEMSpectrum:
                    self.original_spectra = dataset
                    self.spectra = dataset  # load spectra data from bcf file
        # for .hspy files:
        elif file_path.endswith('.hspy'):
            if nag_file_path is not None:
                assert nag_file_path.endswith('.hspy')
                nav_img = hs.load(nag_file_path)
            else:
                nav_img = Signal2D(self.base_dataset.sum(axis=2).data).T
            
            self.original_nav_img = nav_img
            self.nav_img = nav_img
            self.original_spectra = self.base_dataset
            self.spectra = self.base_dataset
            
        #for .rpl files from AZTEC
        elif file_path.endswith('.rpl'):
            self.base_dataset=self.base_dataset.transpose()
            
            #setting up the units as the rpl file has none
            self.base_dataset.axes_manager[2].name='Energy'
            self.base_dataset.axes_manager[2].units='keV'
            self.base_dataset.axes_manager[0].name='width'
            self.base_dataset.axes_manager[1].name='height'
            
            self.base_dataset.set_signal_type('EDS_SEM')
            
            scale=None
            offset=None
            
            if isfile(file_path[:-4]+'.txt'):
                with open(file_path[:-4]+'.txt') as f:
                    for line in f:
                        if "#FORMAT      : EMSA/MAS Spectral Data File" in line: #check if .txt file is EMSA format
                            for line in f:
                                if line.startswith("#XPERCHAN"):
                                    scale = float(re.findall("(\d+\.\d+)", line)[0]) #using regular expression to extract the float from this line
                                elif line.startswith("#OFFSET"):
                                    offset = float(re.findall("(\d+\.\d+)", line)[0])
                                    if "-" in line:
                                        offset = 0-offset
                            break
                        
            
            else:
                print("could not find EMSA/MAS file assuming scale of 0.01keV /Ch and offset of -0.2 keV")
                self.base_dataset.axes_manager[2].scale=0.01
                self.base_dataset.axes_manager[2].offset=-0.2
            
            if not isfile(file_path[:-4]+'.par'):
                print("Could not find .par file - no scaling applied to x/y axis")
            else:
                print('reading x/y scaling parameters from '+file_path[:-4]+'.par')
                params=read_par(file_path[:-4]+'.par')
                
                self.base_dataset.axes_manager[0].scale=params['scale_x']
                self.base_dataset.axes_manager[1].scale=params['scale_y']
                
                self.base_dataset.axes_manager[0].units='μm'
                self.base_dataset.axes_manager[1].units='μm'
                
                if scale and offset:
                    print("reading energy scale and offset from EMSA/MAS file")
                    self.base_dataset.axes_manager[-1].scale=scale
                    self.base_dataset.axes_manager[-1].offset=offset
                    
                
            
            sum_image=self.base_dataset.sum(axis=2)
            if nag_file_path is not None:
                
                nav_img = hs.load(nag_file_path)
                # Convert RGB (structured or regular) to grayscale if needed
                if nav_img.data.dtype.names is not None:
                    rgb = np.stack([nav_img.data[name] for name in ('R', 'G', 'B')], axis=-1)
                    nav_img.data = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
                elif nav_img.data.ndim == 3 and nav_img.data.shape[-1] == 3:
                    nav_img.data = np.dot(nav_img.data[..., :3], [0.2989, 0.5870, 0.1140])
                    
                target_shape = sum_image.data.shape[-2:]  # (height, width)
                if nav_img.data.shape != target_shape:
                    nav_img.data = resize(nav_img.data, target_shape, preserve_range=True, anti_aliasing=True)
                    nav_img.data = nav_img.data.astype(np.float32)
                # Sync axes info
                nav_img.axes_manager[0].units = sum_image.axes_manager[0].units
                nav_img.axes_manager[1].units = sum_image.axes_manager[1].units
                nav_img.axes_manager[0].scale = (
                    sum_image.axes_manager[0].scale * (sum_image.axes_manager[0].size / nav_img.axes_manager[0].size)
                )
                nav_img.axes_manager[1].scale = (
                    sum_image.axes_manager[1].scale * (sum_image.axes_manager[1].size / nav_img.axes_manager[1].size)
                )

        # Final shared logic
        self.spectra.change_dtype("float32")
        self.spectra_raw = self.spectra.deepcopy()
        self.spectra_raw.metadata.add_dictionary(self.spectra.metadata.as_dictionary())
        self.spectra_uncalibrated = self.spectra.deepcopy()

        try:
            self.feature_list = self.spectra.metadata.Sample.xray_lines
            self.feature_dict = {el: i for i, el in enumerate(self.feature_list)}
        except AttributeError:
            print("No X-ray line metadata found — defaulting to empty")
            self.spectra.metadata.set_item('Sample.xray_lines', [])
            
            
    def calibrate_spectra(self, measured_peaks_dict=None, tolerance=0.2, reset=True):
        # Reference line library
        reference_library = {
            'C_Ka': 0.277, 'N_Ka': 0.392, 'O_Ka': 0.525, 'F_Ka': 0.677, 'Na_Ka': 1.041,
            'Mg_Ka': 1.253, 'Al_Ka': 1.486, 'Si_Ka': 1.740, 'P_Ka': 2.013, 'S_Ka': 2.308,
            'Cl_Ka': 2.622, 'K_Ka': 3.312, 'Ca_Ka': 3.690, 'Ti_Ka': 4.510, 'Mn_Ka': 5.895,
            'Fe_Ka': 6.404, 'Co_Ka': 6.930, 'Ni_Ka': 7.470, 'Cu_Ka': 8.040, 'Zn_Ka': 8.640,
        }
        if reset:
        
            self.spectra.data[:] = self.spectra_uncalibrated.data.copy()
            self.spectra.axes_manager[2].scale = self.spectra_uncalibrated.axes_manager[2].scale
            self.spectra.axes_manager[2].offset = self.spectra_uncalibrated.axes_manager[2].offset
            # self.spectra.metadata.add_dictionary(self.spectra_uncalibrated.metadata.as_dictionary())
        if measured_peaks_dict is not None:
            matched_measured = []
            matched_reference = []
            for line, measured_energy in measured_peaks_dict.items():
                if line not in reference_library:
                    raise ValueError(f"Unknown line label '{line}'")
                matched_measured.append(measured_energy)
                matched_reference.append(reference_library[line])
                print(f"[Manual] {line}: Measured = {measured_energy:.3f} keV → Ref = {reference_library[line]:.3f} keV")
            matched_measured = np.array(matched_measured)
            matched_reference = np.array(matched_reference)
        else:
            energy = self.spectra.axes_manager[2].axis
            counts = self.spectra.sum().data
            peaks_idx, _ = find_peaks(counts, height=0.05 * np.max(counts), distance=5)
            measured_peaks = energy[peaks_idx]
            reference_lines = np.array(list(reference_library.values()))
            matched_measured, matched_reference = match_peaks(measured_peaks, reference_lines, tolerance=tolerance)
        if len(matched_measured) < 2:
            raise ValueError("Not enough matched peaks for calibration.")
        a, b = np.polyfit(matched_measured, matched_reference, 1)
        print(f"Calibration: E_corrected = {a:.6f} * E_measured + {b:.6f}")
        self.spectra.axes_manager[2].scale *= a
        self.spectra.axes_manager[2].offset = self.spectra.axes_manager[2].offset * a + b
        print("Calibration successful.")
        
        
    def get_feature_maps_with_nav_img(self, normalisation: Union[str, list, Callable] = "zscore"):
        """
        Returns combined feature maps including the navigation image, with optional normalisation.
        
        Parameters
        ----------
        normalisation : str, list of str, or callable
            Normalisation method(s) for the nav image before combining.
        """
        if self.normalised_elemental_data is not None:
            feature_maps = self.normalised_elemental_data
            if not normalisation:
                print('WARNING - no normalisation parameters passed, but the dataset is normalised. Clustering may be adversely affected')
            
        else:
            feature_maps = self.get_feature_maps(self.feature_list)  # shape: [x, y, n_features]
        if self.nav_img is None:
            raise ValueError("Navigation image not loaded. Call add_nav_img_to_feature_list() first.")
        nav_img = self.nav_img.data
        # Resize nav image if shape mismatch
        if nav_img.shape != feature_maps.shape[:2]:
            print(f"Resizing nav_img from {nav_img.shape} to {feature_maps.shape[:2]}")
            nav_img = resize(nav_img, feature_maps.shape[:2], preserve_range=True, anti_aliasing=True).astype(np.float32)
        # Apply normalisation
        if isinstance(normalisation, str) or callable(normalisation):
            normalisation = [normalisation]
        for norm_step in normalisation:
            if callable(norm_step):
                nav_img = norm_step(nav_img[..., np.newaxis])[:, :, 0]
            elif norm_step in [None, "none"]:
                pass
            else:
                raise ValueError(f"Unknown normalisation method: {norm_step}")
        # Expand nav_img to match feature dimension and combine
        nav_img_expanded = nav_img[..., np.newaxis]  # shape: [x, y, 1]
        combined = np.concatenate([feature_maps, nav_img_expanded], axis=-1)
        
        if self.normalised_elemental_data is not None:
            self.normalised_elemental_data=combined
        else:
            self.features_with_nav_img = combined
        self.feature_list = [el for el in self.feature_list if el != "Navigator"]
        self.feature_list.append("Navigator")
        
        self.nav_img_feature = self.nav_img
    
    def _load_h5oina(self, file_path):
            with h5py.File(file_path, 'r') as f:
                ### --- Load SE Image --- ###
                try:
                    se_data = f['1/Electron Image/Data/SE']
                    se_name = list(se_data.keys())[0]  # handles arbitrary dataset names
                    se_dataset = se_data[se_name]
                    
                    x_se = int(f['1/Electron Image/Header/X Cells'][0])
                    y_se = int(f['1/Electron Image/Header/Y Cells'][0])
                    x_step_se = float(f['1/Electron Image/Header/X Step'][0])
                    y_step_se = float(f['1/Electron Image/Header/Y Step'][0])

                    se_img_flat = se_dataset[()]
                    se_img = se_img_flat.reshape((y_se, x_se))  # (rows, cols)

                    self.se_signal = hs.signals.Signal2D(se_img.astype(np.float32))
                    self.se_signal.axes_manager[0].scale = y_step_se
                    self.se_signal.axes_manager[1].scale = x_step_se
                    self.se_signal.metadata.General.title = "SE Image"
                    
                    self.nav_img=self.se_signal
                    self.nav_img.axes_manager[0].units='μm'
                    self.nav_img.axes_manager[1].units='μm'
                except Exception as e:
                    print(f"Warning: SE image loading failed: {e}")

                ### --- Load EDS Spectrum Map --- ###
                try:
                    spectrum = f['1/EDS/Data/Spectrum'][()]
                    x_eds = int(f['1/EDS/Header/X Cells'][0])
                    y_eds = int(f['1/EDS/Header/Y Cells'][0])
                    start_energy = float(f['1/EDS/Header/Start Channel'][0])
                    channel_width = float(f['1/EDS/Header/Channel Width'][0])

                    spectrum_reshaped = spectrum.reshape((y_eds, x_eds, -1))
                    self.spectra = hs.signals.EDSSEMSpectrum(spectrum_reshaped)
                    
                    
                    # Axes setup
                    self.spectra.axes_manager[0].name = 'y'
                    self.spectra.axes_manager[1].name = 'x'
                    self.spectra.axes_manager[2].name = 'Energy'
                    self.spectra.axes_manager[2].scale = channel_width/1000 #factor of 1000 to convert to keV
                    self.spectra.axes_manager[2].offset = start_energy/1000
                    self.spectra.metadata.General.title = "EDS Spectrum Map"
                    self.spectra.axes_manager[2].units='keV'
                except Exception as e:
                    print(f"Warning: EDS spectrum loading failed: {e}")
                    
                # Resize nav_img to match spectra shape if needed
                try:
                    target_shape = self.spectra.data.shape[:2]  # (y, x)
                    if self.nav_img.data.shape != target_shape:
                        print(f"Resizing nav_img from {self.nav_img.data.shape} to {target_shape}")
                        self.nav_img.data = resize(
                            self.nav_img.data,
                            target_shape,
                            preserve_range=True,
                            anti_aliasing=True
                        ).astype(np.float32)
                except Exception as e:
                    print(f"Warning: nav_img resizing failed: {e}")

class IMAGEDataset(object):
    def __init__(self, 
                 chemical_maps_dir: Union[str, Path], 
                 intensity_map_path: Union[str, Path]
                 ):

        chemical_maps_paths = [join(chemical_maps_dir, f) for f in os.listdir(chemical_maps_dir) if not f.startswith('.')]
        chemical_maps = [Image.open(p) for p in chemical_maps_paths]
        chemical_maps = [ImageOps.grayscale(p) for p in chemical_maps]
        chemical_maps = [np.asarray(img) for img in chemical_maps]

        self.chemical_maps = np.stack(chemical_maps,axis=2).astype(np.float32)
        self.intensity_map = np.asarray(Image.open(intensity_map_path).convert('L')).astype(np.int32)

        self.chemical_maps_bin = None
        self.intensity_map_bin = None
        
        self.feature_list = [f.split('.')[0] for f in os.listdir(chemical_maps_dir) if not f.startswith('.')]
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        
    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set feature_list to {self.feature_list}")
    
    def rebin_signal(self, size:Tuple=(2,2)):
        for (i, maps) in enumerate([self.chemical_maps, self.intensity_map]):
            w, h = maps.shape[:2]
            new_w, new_h = int(w/size[0]), int(h/size[1])
            maps = resize(maps, (new_w, new_h))
            if i ==0: 
                self.chemical_maps_bin = maps
            else: 
                self.intensity_map_bin = maps

    def normalisation(self, norm_list:List=[]):
        self.normalised_elemental_data = self.chemical_maps_bin if self.chemical_maps_bin is not None else self.chemical_maps
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i+1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )

class PIXLDataset(IMAGEDataset):
    def __init__(self, file_path: Union[str, Path]):
        self.base_dataset = hs.load(file_path)
        self.chemical_maps = self.base_dataset.data.astype(np.float32)
        self.intensity_map = self.base_dataset.data.sum(axis=2).astype(np.float32)
        self.intensity_map = self.intensity_map / self.intensity_map.max()
        
        self.chemical_maps_bin = None
        self.intensity_map_bin = None
        
        self.feature_list = self.base_dataset.metadata.Signal.phases
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}


class AZTECDataset(object):
    """
    Object, similar to BaseDataSet, but does not initialise from a file path, instead intitialises from a hypserpsy signal
    """
    
    def __init__(self, hs_signal):
        self.base_dataset = hs_signal
        self.nav_img = None
        self.spectra = None
        self.original_nav_img = None
        self.original_spectra = None
        self.nav_img_bin = None
        self.spectra_bin = None
        self.spectra_raw = None
        self.feature_list = []
        self.feature_dict = {}

    def set_feature_list(self, feature_list):
        """
        Sets the feautre_list attribute of the BaseDataSet to the defined feature list.
        Addst this feature list to the metadata of the spectra attribute.
        Creates a feature_dict attributre of the Basedataset containing the X-Ray lines in feature list.

        
        Parameters
        ----------
        feature_list : List
            A list containing all of the features to add to .feature_list, eg [Cu_Ka,O_Ka...]
            
        """
        self.feature_list = feature_list
        for s in [self.spectra, self.spectra_bin]:
            if s is not None:
                s.metadata.Sample.xray_lines = self.feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(feature_list)}
        print(f"Set feature_list to {self.feature_list}")

    def rebin_signal(self, size=(2, 2)):
        """
        Rebins the navigation axes of the hyperspectral image and navigation image (eg. a BSE image) contained in BaseDataSet.
        

        
        Parameters
        ----------
        size : Tuple
            A 2 element tuple of the form (x_bin,y_bin) where x_bin and y_bin define the number of pixels to sum in the x and y axis respectively
            into a single, summed pixel in the binned signal.

        Returns
        ----------
        (spectra_bin,nav_img_bin) : Tuple
            The binned signals of the hyperspectral image and the navigation image
            
        """
        print(f"Rebinning the intensity with the size of {size}")
        x, y = size[0], size[1]
        
        #getting the dimensions of the binned array
        new_x=int(self.spectra.shape[0]/x)
        new_y=int(self.spectra.shape[1]/y)
        
        self.spectra_bin = bin_ndarray(self.spectra,(new_x,new_y,self.spectra.shape[2]))
        self.nav_img_bin = self.nav_img.rebin(scale=(x, y))
        self.spectra_raw = copy.deepcopy(self.spectra_bin)
        return (self.spectra_bin, self.nav_img_bin)

    def remove_first_peak(self, end: float):
        """
        Removes the zero energy peak from the spectrum, by removing cropping the signal axis so that it begins at an energy defined by the user
        

        
        Parameters
        ----------
        end : float
            Energy, in keV, after which the signal is retained (ie. everything up to 'end' is cropped out of the signal
            
        """
        
        print(
            f"Removing the first peak by setting the intensity to zero until the energy of {end} keV."
        )
        for spectra in (self.spectra, self.spectra_bin):
            if spectra is None:
                continue
            else:
                scale = spectra.axes_manager[2].scale
                offset = spectra.axes_manager[2].offset
                end_ = int((end - offset) / scale)
                for i in range(end_):
                    spectra.isig[i] = 0

    def peak_intensity_normalisation(self) -> EDSSEMSpectrum:
        """
        Normalises the integrated intensity of the EDS signal, so that the sum along the signal axis is 1.
            
        """
        print(
            "Normalising the chemical intensity along axis=2, so that the sum is equal to 1 along axis=2."
        )
        if self.spectra_bin:
            spectra_norm = self.spectra_bin
        else:
            spectra_norm = self.spectra
        spectra_norm.data = spectra_norm.data / spectra_norm.data.sum(axis=2, keepdims=True)
        if np.isnan(np.sum(spectra_norm.data)):
            spectra_norm.data = np.nan_to_num(spectra_norm.data)
        return spectra_norm

    def peak_denoising_PCA(
        self, n_components_to_reconstruct=10, plot_results=True
    ) -> EDSSEMSpectrum:
        print("Peak denoising using PCA.")
        if self.spectra_bin:
            spectra_denoised = self.spectra_bin
        else:
            spectra_denoised = self.spectra
        spectra_denoised.decomposition(
            normalize_poissonian_noise=True,
            algorithm="SVD",
            random_state=0,
            output_dimension=n_components_to_reconstruct,
        )

        if plot_results == True:
            spectra_denoised.plot_decomposition_results()
            spectra_denoised.plot_explained_variance_ratio(log=True)
            spectra_denoised.plot_decomposition_factors(comp_ids=4)

        return spectra_denoised

    def get_feature_maps(self, feature_list=None) -> np.ndarray:
        """
        Produces elemental / feature maps for the EmptyDataset object, that was initialised with a hyperspy.signal1d object where the signal dimension is the intensity of each elemental map.

		Parameters
		----------
		feature_list : string
					   list of X-Ray lines to create elemental maps for


		Returns
		-------
		data_cube : 3D numpy array
					3D data cube, containing both navigation dimensions and an intensity for each feature in feature list.
					The dimensions are: (length of x axis) x (length of y axis) x (number of feautures)

        """
        
        data_cube=self.base_dataset.data 
        

        return data_cube

    def normalisation(self, norm_list=[]):
        self.normalised_elemental_data = self.get_feature_maps(self.feature_list)
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i+1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )
	


import h5py

def clean_metadata(md):
    """
    Function for cleaning metadata, useful for reading .h5oina files.
    
    Removes any 'bytes' parts from the metadata so it can be saved as hdf5 file
    
    Pararameters
    ------------
    md : dictionary, containing metadata from a h5oina file
    
    Returns
    --------
    cleaned : dictionary, contains the cleaned metadata
    """
    
    cleaned = {}
    for k, v in md.items():
        if isinstance(v, bytes):
            try:
                v = v.decode('utf-8')
            except UnicodeDecodeError:
                continue  # skip if undecodable
        elif isinstance(v, np.ndarray):
            try:
                v = v.item()  # convert 0-d arrays to scalars
            except:
                continue  # skip if it fails
        elif isinstance(v, (dict, list, tuple)):
            continue  # skip complex types
        cleaned[k] = v
    return cleaned
 
 
 
#rebin function needed for binning maps of AZTECDataset
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if operation not in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray
    

 
def load_AZTEC(input_file,y_dim=1024):
    """
    Function for loading a .h5oina file to a BaseDataset objet
    
    Parameters
    -----------
    input_file : str
                 path pointing to the input .h5oina file
               
    y_dim : int, default=1024. The number of pixels in the actual y axis. Needed if the plot is read (by deafault) as a 1D linescan like object.
    
    
    Returns
    -------
    Aztec_data : BaseDataset like object, though will not have a signal axis. Should have a nav_img and elemental_maps objects.
    
    
    """
    # === Open file ===
    with h5py.File(input_file, 'r') as f:
        win_int_path = "1/EDS/Data/Window Integral"
        element_names = list(f[win_int_path].keys())
        
        # Load maps
        maps = [f[f"{win_int_path}/{el}"][:] for el in element_names]
        data = np.stack(maps, axis=0)  # Shape: (elements, Y, X)
        
        # Transpose to match HyperSpy: (Y, X, elements)
        data = np.moveaxis(data, 0, -1)

        # Axis scaling from metadata
        x_scale = f["1/EDS/Header/X Step"][()]
        y_scale = f["1/EDS/Header/Y Step"][()]

        # Metadata
        metadata = {}
        for key in f["1/EDS/Header"].keys():
            try:
                val = f[f"1/EDS/Header/{key}"][()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                metadata[key] = val
            except:
                pass
                
    # === Create HyperSpy signal ===
    s_tmp = hs.signals.Signal2D(data,dtype='float32') # creating a temporary file that contains all of the data but is the wrong shape
    s=hs.signals.Signal1D(s_tmp.data.reshape(int(len(s_tmp.as_signal1D(spectral_axis=0).data)/y_dim),y_dim,len(maps)),dtype='float32')
    del(s_tmp) #removing the temporary variable from memory 
    #creating an AZtec dataset Signal object
    aztec_data=AZTECDataset(s)
    #creating the navigation image
    aztec_data.nav_img=s.sum(axis=2) 
   
    #creating the feature_list
    aztec_data.feature_list=element_names
    
    aztec_data.spectra=s.data
    return aztec_data
    
    
import re

def parse_length(line):
    """Extract a length value and convert to microns (μm)."""
    # Convert units to microns (μm)
    unit_factors_to_um = {
        'km': 1e9,       # 1 km = 1e9 μm
        'm': 1e6,
        'cm': 1e4,
        'mm': 1e3,
        'μm': 1.0,
        'um': 1.0,       # ASCII version of μm
        'nm': 1e-3,
        'pm': 1e-6,
    }
    match = re.search(r"([\d\.]+)\s*(km|m|cm|mm|μm|um|nm|pm)", line)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value * unit_factors_to_um[unit]
    return None
        
def read_par(filepath):
    """
    Helper function to read a .par file and extract spatial and spectral scale parameters.

    Parameters
    ----------
    filepath : str
        Path to the .par file.
    
    Returns
    -------
    params : dict
        Dictionary of the form:
        {
            'scale_x': float,  # microns per pixel
            'scale_y': float,  # microns per pixel
            'scale_spectral': float,  # keV per channel
        }
    """
    params = {}
    width_um = height_um = None
    width_px = height_px = None
    energy_range = num_channels = None





    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if "Map Width" in line:
                width_um = parse_length(line)
            elif "Map Height" in line:
                height_um = parse_length(line)
            elif "Resolution (Width)" in line:
                match = re.search(r"(\d+)\s*pixels", line)
                if match:
                    width_px = int(match.group(1))
            elif "Resolution (Height)" in line:
                match = re.search(r"(\d+)\s*pixels", line)
                if match:
                    height_px = int(match.group(1))
            elif "Energy Range" in line:
                match = re.search(r"([\d\.]+)\s*keV", line)
                if match:
                    energy_range = float(match.group(1))
            elif "Number Of Channels" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    num_channels = int(match.group(1))

    # Compute scales
    try:
        params['scale_x'] = width_um / width_px
        params['scale_y'] = height_um / height_px
        params['scale_spectral'] = energy_range / num_channels
    except TypeError:
        raise ValueError("Missing or invalid parameters in the file.")

    return params
    



from scipy.optimize import linear_sum_assignment


def match_peaks(measured, reference, tolerance=0.2):
    
  
    # Full cost matrix
    cost_matrix = np.abs(measured[:, np.newaxis] - reference[np.newaxis, :])

    # Mask elements beyond the tolerance
    valid_mask = cost_matrix <= tolerance

    # Identify valid measured and reference indices (with at least one match)
    valid_measured_idx = np.any(valid_mask, axis=1)
    valid_reference_idx = np.any(valid_mask, axis=0)

    # If not enough peaks, exit early
    if np.sum(valid_measured_idx) < 2 or np.sum(valid_reference_idx) < 2:
        raise ValueError("Not enough close matches within tolerance to calibrate.")

    # Filter down cost matrix and values
    reduced_cost_matrix = cost_matrix[np.ix_(valid_measured_idx, valid_reference_idx)]

    # Set inf where outside tolerance (still needed to avoid bad matches)
    reduced_cost_matrix[reduced_cost_matrix > tolerance] = np.inf

    # Now apply linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(reduced_cost_matrix)

    # Filter out invalid matches (still needed if infs remain)
    matched_costs = reduced_cost_matrix[row_ind, col_ind]
    valid = matched_costs < np.inf

    # Recover original indices
    matched_measured = measured[valid_measured_idx][row_ind[valid]]
    matched_reference = reference[valid_reference_idx][col_ind[valid]]

    return matched_measured, matched_reference

# ---- offline-friendly EMD loader for HS 1.7 with pre-warmed HS 2.x bridge ----
# Keep this file in your HS 1.7 environment.
import os, sys, json, shutil, subprocess, tempfile, pathlib, platform, time
from typing import Tuple, Optional, List

import numpy as np
import h5py
import hyperspy.api as hs

# ---------------- helpers: quick pruned detection ----------------
def _emd_si_is_pruned(emd_path: str) -> bool:
    with h5py.File(emd_path, "r") as f:
        if "Data/SpectrumImage" not in f:
            return False
        for _, grp in f["Data/SpectrumImage"].items():
            if not isinstance(grp, h5py.Group) or "Data" not in grp:
                continue
            ds = grp["Data"]
            shp = tuple(ds.shape)
            # hallmark: smallish uint8 blob, no Dimensions or mismatched dims
            if ds.dtype == np.uint8 and (len(shp) in (1,2)) and (len(shp)==1 or shp[-1]==1):
                if "Dimension" not in grp:
                    return True
                try:
                    dims = grp["Dimension"]
                    idxs = [int(k) for k in dims.keys() if k.isdigit()]
                    lens = []
                    for i in sorted(idxs):
                        d = dims[str(i)]
                        if "Length" not in d: break
                        lens.append(int(d["Length"][()]))
                    if lens and int(np.prod(lens)) == int(np.prod(shp)):
                        return False
                except Exception:
                    pass
                return True
    return False

# -------------- HS 1.7 direct loader for non-pruned files --------------
def _load_with_hs17_direct(emd_path: str):
    try:
        objs = hs.load(emd_path, lazy=False)
    except Exception:
        objs = []
    if not isinstance(objs, (list, tuple)):
        objs = [objs] if objs else []
    return objs

# -------------- private venv manager (bridge) --------------
def _cache_root() -> pathlib.Path:
    base = os.getenv("XDG_CACHE_HOME") or os.path.join(pathlib.Path.home(), ".cache")
    return pathlib.Path(base) / "hs2_bridge"

def _venv_dir_for_py() -> pathlib.Path:
    return _cache_root() / f"venv_py{sys.version_info.major}{sys.version_info.minor}"

def _venv_python(venv_dir: pathlib.Path) -> pathlib.Path:
    return venv_dir / ("Scripts/python.exe" if platform.system().lower().startswith("win") else "bin/python")

# ---- Bridge venv maintenance: ensure required packages even if venv already exists ----
import os, sys, subprocess, platform, pathlib

def _cache_root() -> pathlib.Path:
    base = os.getenv("XDG_CACHE_HOME") or os.path.join(pathlib.Path.home(), ".cache")
    return pathlib.Path(base) / "hs2_bridge"

def _venv_dir_for_py() -> pathlib.Path:
    return _cache_root() / f"venv_py{sys.version_info.major}{sys.version_info.minor}"

def _venv_python(venv_dir: pathlib.Path) -> pathlib.Path:
    return venv_dir / ("Scripts/python.exe" if platform.system().lower().startswith("win") else "bin/python")

def _pip_run(py_exe: pathlib.Path, args, env=None):
    cmd = [str(py_exe), "-m", "pip"] + list(args)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def _bridge_has(py_exe: pathlib.Path, pkg: str) -> bool:
    out = _pip_run(py_exe, ["show", pkg])
    return out.returncode == 0

def _ensure_bridge_packages(py_exe: pathlib.Path, packages, wheelhouse=None, offline=False):
    # build pip install command
    base = []
    if wheelhouse:
        wheelhouse = str(pathlib.Path(wheelhouse).resolve())
        if offline:
            base += ["--no-index", f"--find-links={wheelhouse}"]
        else:
            base += [f"--find-links={wheelhouse}"]  # prefer local, allow internet
    elif offline:
        raise RuntimeError("offline=True but no wheelhouse provided")

    missing = [p for p in packages if not _bridge_has(py_exe, p.split("==")[0].split(">=")[0])]
    if not missing:
        return
    # upgrade pip first (safe even offline if wheel in wheelhouse)
    _pip_run(py_exe, ["install", "--upgrade", "pip", "setuptools", "wheel"] + ([] if not base else base))
    # install missing
    res = _pip_run(py_exe, ["install"] + (base if base else []) + missing)
    if res.returncode != 0:
        raise RuntimeError(f"[hs2-bridge] failed to install {missing}:\n{res.stdout}")

def prime_hs2_bridge(wheelhouse: str | None = None, offline: bool = False,
                     quiet: bool = False, force_recreate: bool = False) -> pathlib.Path:
    """
    Ensure the private venv exists AND has required packages.
    Returns the path to venv's python.
    - wheelhouse: folder with pre-downloaded wheels (for offline)
    - offline: install only from wheelhouse if True
    - force_recreate: delete and recreate the venv
    """
    import venv as _venv

    venv_dir = _venv_dir_for_py()
    py = _venv_python(venv_dir)

    if force_recreate and venv_dir.exists():
        if not quiet: print(f"[hs2-bridge] removing venv: {venv_dir}")
        import shutil; shutil.rmtree(venv_dir, ignore_errors=True)

    if not py.exists():
        if not quiet: print(f"[hs2-bridge] creating venv at {venv_dir}")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _venv.EnvBuilder(with_pip=True, clear=True).create(str(venv_dir))
        # minimal bootstrap
        _pip_run(py, ["install", "--upgrade", "pip", "setuptools", "wheel"])

    else:
        if not quiet: print(f"[hs2-bridge] using existing venv: {venv_dir}")

    # Always ensure required packages (even on existing venv)
    required = ["hyperspy>=2", "rosettasciio", "sparse", "numba"]
    _ensure_bridge_packages(py, required, wheelhouse=wheelhouse, offline=offline)

    return py

# If you also call the converter, keep the backend-safe env when spawning it:
def _convert_with_hs2(emd_path: str, out_dir: pathlib.Path, py_bridge: pathlib.Path) -> list:
    _CONVERTER_SNIPPET = r"""
import os
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import sys, json, pathlib
import hyperspy.api as hs

in_path = pathlib.Path(sys.argv[1]).resolve()
out_dir = pathlib.Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

objs = hs.load(str(in_path), lazy=False)
if not isinstance(objs, (list, tuple)):
    objs = [objs]
out = []
for i, s in enumerate(objs):
    title = (s.metadata.General.title or f"signal_{i}").replace("/", "-")
    dst = out_dir / f"{i:02d}_{title}.hspy"
    s.save(str(dst), overwrite=True)
    out.append(dict(index=i, title=str(s.metadata.General.title or ""), path=str(dst)))
print(json.dumps(out))
"""
    conv_py = out_dir / "hs2_convert.py"
    conv_py.write_text(_CONVERTER_SNIPPET, encoding="utf-8")
    cmd = [str(py_bridge), str(conv_py), str(pathlib.Path(emd_path).resolve()), str(out_dir)]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"[hs2-bridge] converter failed:\n{proc.stdout}")
    # parse last JSON line
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    import json
    for ln in reversed(lines):
        if ln.lstrip().startswith(("{","[")):
            return json.loads(ln)
    raise RuntimeError(f"[hs2-bridge] unexpected converter output:\n{proc.stdout}")

# Helper to pre-download wheels (run once online if you want a local wheelhouse)
def download_bridge_wheels(target_dir: str):
    """
    Download wheels for hyperspy>=2 and rosettasciio (and dependencies) for your platform & Python.
    Run this ONCE while online; later you can prime the bridge with offline=True.
    """
    tgt = pathlib.Path(target_dir).resolve()
    tgt.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--dest", str(tgt), "hyperspy>=2", "rosettasciio"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"[wheelhouse] download failed:\n{proc.stdout}")
    print(f"[wheelhouse] wheels saved to {tgt}")

# -------------- converter subprocess (uses the bridge) --------------
_CONVERTER_SNIPPET = r"""
import os
# Force headless matplotlib so Hyperspy can import cleanly in a non-notebook subprocess
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Optional: isolate matplotlib config (avoids permission issues on locked systems)
os.environ.setdefault("MPLCONFIGDIR", str((__import__('pathlib').Path.cwd() / 'mplconfig').resolve()))

import sys, json, pathlib
import hyperspy.api as hs

in_path = pathlib.Path(sys.argv[1]).resolve()
out_dir = pathlib.Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

objs = hs.load(str(in_path), lazy=False)
if not isinstance(objs, (list, tuple)):
    objs = [objs]

out = []
for i, s in enumerate(objs):
    title = (s.metadata.General.title or f"signal_{i}").replace("/", "-")
    dst = out_dir / f"{i:02d}_{title}.hspy"
    s.save(str(dst), overwrite=True)
    out.append(dict(index=i, title=str(s.metadata.General.title or ""), path=str(dst)))
print(json.dumps(out))
"""

def _convert_with_hs2(emd_path: str, out_dir: pathlib.Path, py_bridge: pathlib.Path) -> list:
    conv_py = out_dir / "hs2_convert.py"
    conv_py.write_text(_CONVERTER_SNIPPET, encoding="utf-8")
    cmd = [str(py_bridge), str(conv_py), str(pathlib.Path(emd_path).resolve()), str(out_dir)]

    # Clean environment for the subprocess so notebook-specific backends don't leak in
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.pop("PYTHONPATH", None)  # avoid pulling in the parent env’s site-packages by accident

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"[hs2-bridge] converter failed:\n{proc.stdout}")

    # Find the JSON line robustly (ignore any print noise)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") or ln.startswith("["):
            try:
                return json.loads(ln)
            except Exception:
                break
    raise RuntimeError(f"[hs2-bridge] unexpected converter output:\n{proc.stdout}")

# -------------- cache helpers --------------
def _default_cache_for(emd_path: str) -> pathlib.Path:
    # sidecar cache folder next to the .emd (e.g., myfile.emd.cache/)
    p = pathlib.Path(emd_path).resolve()
    return p.with_suffix(p.suffix + ".cache")

def _is_cache_fresh(emd_path: str, cache_dir: pathlib.Path) -> bool:
    if not cache_dir.exists(): return False
    emd_mtime = pathlib.Path(emd_path).stat().st_mtime
    marker = cache_dir / ".src_mtime"
    if not marker.exists(): return False
    try:
        saved = float(marker.read_text().strip())
    except Exception:
        return False
    return abs(saved - emd_mtime) < 1e-6

def _mark_cache(emd_path: str, cache_dir: pathlib.Path):
    emd_mtime = pathlib.Path(emd_path).stat().st_mtime
    (cache_dir / ".src_mtime").write_text(str(emd_mtime), encoding="utf-8")

def _load_hspy_folder(cache_dir: pathlib.Path) -> List[hs.signals.BaseSignal]:
    out = []
    for p in sorted(cache_dir.glob("*.hspy")):
        obj = hs.load(str(p), lazy=False)
        if isinstance(obj, list):
            out.extend(obj)
        else:
            out.append(obj)
    return out

# -------------- public API --------------
def load_emd_any(emd_path: str,
                 prefer_bse_order=("HAADF","ADF","DF"),
                 wheelhouse: Optional[str] = None,
                 offline_bridge: bool = False,
                 use_cache: bool = True,
                 cache_dir: Optional[str] = None) -> Tuple[hs.signals.Signal2D, hs.signals.BaseSignal]:
    """
    One-liner loader that returns (BSE image, EDS spectrum image) using HS 1.7.
    - If EMD is pruned, uses a pre-warmed HS 2.x bridge (can be fully offline if primed).
    - Caches converted .hspy files to avoid re-conversion.
    Params:
      wheelhouse: path with pre-downloaded wheels (for offline bridge priming)
      offline_bridge: True to force bridge install from wheelhouse only
      use_cache: reuse .hspy cache if up-to-date
      cache_dir: custom cache folder; defaults to '<file>.emd.cache' beside the .emd
    """
    emd_path = os.path.abspath(emd_path)
    cache_dir_path = pathlib.Path(cache_dir) if cache_dir else _default_cache_for(emd_path)

    # 1) Try HS 1.7 directly
    objs = _load_with_hs17_direct(emd_path)

    def _pick(objs_list):
        bse, si = None, None
        for s in objs_list:
            if getattr(s.axes_manager, "signal_dimension", 0) == 2:
                title = (s.metadata.General.title or "").upper()
                rank = {name: i for i, name in enumerate(prefer_bse_order)}
                for key in rank:
                    if key in title and bse is None:
                        bse = s; break
                if bse is None: bse = s
            if getattr(s.axes_manager, "signal_dimension", 0) == 1 and getattr(s.data, "ndim", 0) == 3:
                if getattr(s.metadata.Signal, "signal_type", "") in ("EDS_TEM","EDS_SEM","EDS"):
                    si = s
        return bse, si

    bse, si = _pick(objs)
    if bse is not None and si is not None:
        return bse, si

    # 2) If pruned or SI missing, use cache/bridge
    if use_cache and _is_cache_fresh(emd_path, cache_dir_path):
        loaded = _load_hspy_folder(cache_dir_path)
        bse2, si2 = _pick(loaded)
        if bse2 and si2:
            return bse2, si2

    # If file is pruned or we just couldn't get SI: convert via bridge
    if _emd_si_is_pruned(emd_path) or si is None:
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        py_bridge = prime_hs2_bridge(wheelhouse=wheelhouse, offline=offline_bridge, quiet=False)
        converted = _convert_with_hs2(emd_path, cache_dir_path, py_bridge)
        _mark_cache(emd_path, cache_dir_path)
        loaded = _load_hspy_folder(cache_dir_path)
        bse2, si2 = _pick(loaded)
        if bse is None: bse = bse2
        if si is None:  si = si2

    if bse is None or si is None:
        raise RuntimeError("Could not obtain both BSE and EDS SI (even via bridge/cache).")
    return bse, si

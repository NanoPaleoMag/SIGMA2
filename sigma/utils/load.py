import os
import numpy as np
import hyperspy.api as hs

from typing import Union, Tuple, List
from pathlib import Path
from PIL import Image, ImageOps
from os.path import isfile, join
from skimage.transform import resize
from hyperspy._signals.eds_sem import EDSSEMSpectrum
from hyperspy._signals.signal2d import Signal2D
from .base import BaseDataset
from scipy.signal import find_peaks

import copy



class SEMDataset(BaseDataset):
    def __init__(self, file_path: Union[str, Path], nag_file_path: Union[str, Path]=None):
        super().__init__(file_path)

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
            
            if not isfile(file_path[:-4]+'.par'):
                print("could not find .par file - no scaling applied to spatial axis, and assuming scale of 20kV/2048 channels in spectra axis")
                self.base_dataset.axes_manager[2].scale=20/2048
            else:
                print('reading parameters from '+file_path[:-4]+'.par')
                params=read_par(file_path[:-4]+'.par')
                self.base_dataset.axes_manager[2].scale=params['scale_spectral']
                self.base_dataset.axes_manager[0].scale=params['scale_x']
                self.base_dataset.axes_manager[1].scale=params['scale_y']
                
                self.base_dataset.axes_manager[0].units='μm'
                self.base_dataset.axes_manager[1].units='μm'
                
            
            nav_img = self.base_dataset.sum(axis=2)
            
            self.original_nav_img = nav_img
            self.nav_img = nav_img
            self.original_spectra = self.base_dataset
            self.spectra = self.base_dataset
            
            
        
        
        self.spectra.change_dtype("float32")  # change spectra data from unit8 into float32
        
        # reserve a copy of the raw data for quantification
        self.spectra_raw = self.spectra.deepcopy()
        try:
            self.feature_list = self.spectra.metadata.Sample.xray_lines
            self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        except AttributeError:
            print('Unable to read X-Ray lines from sample metadata, setting blank list')
            self.spectra.metadata.set_item('Sample.xray_lines',[])
            
    def calibrate_spectra(self, measured_peaks_dict=None, tolerance=0.2):
        """
        Calibrate EDS spectrum using either:
        - measured_peaks_dict: dict of {line_label: measured_energy}, e.g., {'O_Ka': 0.703, 'Fe_Ka': 6.45}
        - OR fallback automatic matching
        
        The reference energies are looked up from a predefined dictionary of known lines.
        """

        # Reference line database (expand as needed)
        reference_library = {
            'C_Ka': 0.277,
            'N_Ka': 0.392,
            'O_Ka': 0.525,
            'F_Ka': 0.677,
            'Na_Ka': 1.041,
            'Mg_Ka': 1.253,
            'Al_Ka': 1.486,
            'Si_Ka': 1.740,
            'P_Ka': 2.013,
            'S_Ka': 2.308,
            'Cl_Ka': 2.622,
            'K_Ka': 3.312,
            'Ca_Ka': 3.690,
            'Ti_Ka': 4.510,
            'Mn_Ka': 5.895,
            'Fe_Ka': 6.404,
            'Co_Ka': 6.930,
            'Ni_Ka': 7.470,
            'Cu_Ka': 8.040,
            'Zn_Ka': 8.640,
        }

        if measured_peaks_dict is not None:
            matched_measured = []
            matched_reference = []
            for line, measured_energy in measured_peaks_dict.items():
                if line not in reference_library:
                    raise ValueError(f"Unknown line label '{line}'. Please check or update the reference database.")
                ref_energy = reference_library[line]
                print(f"[Manual] Using {line}: Measured={measured_energy:.3f} keV → Reference={ref_energy:.3f} keV")
                matched_measured.append(measured_energy)
                matched_reference.append(ref_energy)

            matched_measured = np.array(matched_measured)
            matched_reference = np.array(matched_reference)
        else:
            # fallback: automatic detection and matching
            energy = self.spectra.axes_manager[2].axis
            counts = self.spectra.sum().data
            peaks_idx, _ = find_peaks(counts, height=0.05 * np.max(counts), distance=5)
            measured_peaks = energy[peaks_idx]

            reference_lines = np.array(list(reference_library.values()))
            matched_measured, matched_reference = match_peaks(measured_peaks, reference_lines, tolerance=tolerance)

        # Check sufficient data
        if len(matched_measured) < 2:
            raise ValueError("Not enough matched peaks for calibration.")

        # Fit and apply correction
        a, b = np.polyfit(matched_measured, matched_reference, 1)
        print(f"Calibration correction: E_corrected = {a:.6f} * E_measured + {b:.6f}")
        self.spectra.axes_manager[2].scale *= a
        self.spectra.axes_manager[2].offset = self.spectra.axes_manager[2].offset * a + b
        print("Calibration successful.")

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

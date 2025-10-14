from typing import List
import hyperspy.api as hs
import numpy as np
from sigma.utils.load import SEMDataset
from hyperspy.signals import Signal2D, Signal1D
from hyperspy._signals.signal2d import Signal2D, Signal2D
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from .base import BaseDataset
from typing import Union, Tuple, List, Callable


# Prefer skimage.transform.resize (good quality). Provide fallback that uses scipy.ndimage.zoom
try:
    from skimage.transform import resize  # type: ignore
except Exception:
    try:
        from scipy.ndimage import zoom as _zoom

        def resize(image, output_shape, preserve_range=True, anti_aliasing=True):
            """
            Minimal replacement for skimage.transform.resize using scipy.ndimage.zoom.
            - image: array-like
            - output_shape: (rows, cols) target shape
            Note: this fallback does not provide anti_aliasing control and uses simple zoom.
            """
            image = np.asarray(image)
            # compute zoom factors for each axis (only 2D supported here)
            if image.ndim == 2:
                in_shape = image.shape
                zoom_factors = (output_shape[0] / in_shape[0], output_shape[1] / in_shape[1])
                return _zoom(image, zoom_factors, order=1)  # bilinear
            elif image.ndim == 3:
                # assume last axis is channel
                in_shape = image.shape[:2]
                zoom_factors = (output_shape[0] / in_shape[0], output_shape[1] / in_shape[1], 1.0)
                return _zoom(image, zoom_factors, order=1)
            else:
                raise ValueError("resize fallback supports 2D or 3D images only")
    except Exception:
        # Very last-resort trivial identity function (won't resize)
        def resize(image, output_shape, preserve_range=True, anti_aliasing=True):
            raise ImportError("Neither skimage.transform.resize nor scipy.ndimage.zoom are available. Install scikit-image or scipy to enable image resizing.")

class TEMDataset(BaseDataset):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        
        self.nav_img_feature = None # setting this as None, so it can be later loaded
        self.base_dataset = hs.load(file_path)
        
        if type(self.base_dataset) == Signal2D:
            self.stem = self.base_dataset
            
        else:
            
            if type(self.base_dataset) == Signal1D:
                self.spectra = hs.load(file_path, signal_type="EDS_TEM")
                self.nav_img = Signal2D(self.spectra.data.sum(axis=2)).transpose()
                
                #setting axis attributes
                
                self.nav_img.axes_manager[0].name=self.base_dataset.axes_manager[0].name
                self.nav_img.axes_manager[1].name=self.base_dataset.axes_manager[1].name
                
                self.nav_img.axes_manager[0].offset=self.base_dataset.axes_manager[0].offset
                self.nav_img.axes_manager[1].offset=self.base_dataset.axes_manager[1].offset
                
                self.nav_img.axes_manager[0].scale=self.base_dataset.axes_manager[0].scale
                self.nav_img.axes_manager[1].scale=self.base_dataset.axes_manager[1].scale

                self.nav_img.axes_manager[0].units=self.base_dataset.axes_manager[0].units
                self.nav_img.axes_manager[1].units=self.base_dataset.axes_manager[1].units                
               
                

                
                
                
                
                self.spectra.change_dtype("float32")
                self.spectra_raw = self.spectra.deepcopy()

                self.spectra.metadata.set_item("Sample.xray_lines", [])
                self.spectra.axes_manager["Energy"].scale = 0.01 * 8.07 / 8.08
                self.spectra.axes_manager["Energy"].offset = -0.01
                self.spectra.axes_manager["Energy"].units = "keV"

                self.feature_list = []
            
            # if data format is .emd file
            elif type(self.base_dataset) is list: #file_path[-4:]=='.emd' and 
                emd_dataset = self.base_dataset
                self.nav_img = None
                for dataset in emd_dataset:
                    if (self.nav_img is None) and (dataset.metadata.General.title == "HAADF"):
                        self.original_nav_img = dataset
                        self.nav_img = dataset  # load HAADF data
                    elif type(dataset) is EDSTEMSpectrum:
                        self.original_spectra = dataset
                        self.spectra = dataset  # load spectra data from .emd file
						
			

                self.spectra.change_dtype("float32")  
                self.spectra_raw = self.spectra.deepcopy()

                elements = self.spectra.metadata.Sample.elements
                self.spectra.metadata.set_item("Sample.xray_lines", [e+'_Ka' for e in elements])
                self.feature_list = self.spectra.metadata.Sample.xray_lines
                self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
				
			#if the dataset is an EDSTEM dataset
            elif type(self.base_dataset) is EDSTEMSpectrum:
                #print('Dataset is EDSTEMSPectrum')
                self.nav_img=self.base_dataset.sum(axis='Energy').as_signal2D(image_axes=('x','y')) #creating a navigation image, intensity of each pixel is integrated intensity of Xrays
                self.spectra=self.base_dataset # by default hyperspy sums over navigation axis
                self.spectra.change_dtype("float32")
                self.spectra_raw = self.spectra.deepcopy()
                self.original_nav_img=self.nav_img.deepcopy()
                self.feature_list = self.spectra.metadata.Sample.xray_lines
                self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

            else:
                print('WARNING - Could not Identify dataset type - some functionality may be missing')

            
        

    def set_xray_lines(self, xray_lines: List[str]):
        """
        Set the X-ray lines for the spectra analysis. 

        Parameters
        ----------
        xray_lines : List
            A list consisting of a series of elemental peaks. For example, ['Fe_Ka', 'O_Ka'].

        """
        self.feature_list = xray_lines
        self.spectra.set_lines(self.feature_list)
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set xray_lines to {self.feature_list}")

    def set_axes_scale(self, scale:float):
        """
        Set the scale for the energy axis. 

        Parameters
        ----------
        scale : float
            The scale of the energy axis. For example, given a data set with 1500 data points corresponding to 0-15 keV, the scale should be set to 0.01.

        """
        self.spectra.axes_manager["Energy"].scale = scale
    
    def set_axes_offset(self, offset:float):
        """
        Set the offset for the energy axis. 

        Parameters
        ----------
        offset : float
            the offset of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].offset = offset

    def set_axes_unit(self, unit:str):
        """
        Set the unit for the energy axis. 

        Parameters
        ----------
        unit : float
            the unit of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].unit = unit
    
    def remove_NaN(self):
        """
        Remove the pixels where no values are stored.
        """
        index_NaN = np.argwhere(np.isnan(self.spectra.data[:,0,0]))[0][0]
        self.nav_img.data = self.nav_img.data[:index_NaN-1,:]
        self.spectra.data = self.spectra.data[:index_NaN-1,:,:]

        if self.nav_img_bin is not None:
            self.nav_img_bin.data = self.nav_img_bin.data[:index_NaN-1,:]
        if self.spectra_bin is not None:
            self.spectra_bin.data = self.spectra_bin.data[:index_NaN-1,:,:]

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

        # Avoid appending "Navigator" more than once
        if "Navigator" not in self.feature_list:
            self.feature_list.append("Navigator")
            
        self.nav_img_feature = self.nav_img

      

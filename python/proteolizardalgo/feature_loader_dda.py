"""Module for loading DDA precursor data

This module is providing a class to load a precursor peptide feature
acquired by data depending acquisition. The loader recognizes monoisotopic mass
and charge of feature as determined during DDA procedure.
"""

from cmath import inf
from scipy.stats import poisson, norm
from scipy.optimize import curve_fit
from proteolizarddata.data import PyTimsDataHandleDDA, TimsFrame
from proteolizardalgo.clustering import cluster_precursors_dbscan
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import

class FeatureLoaderDDA():
    """
    Class to load a precursor peptide feature
    by it's Id from a DDA dataset.
    """

    def __init__(self, data_handle: PyTimsDataHandleDDA, precursor_id: int):
        self.dataset_pointer = data_handle
        self.precursor_id = precursor_id
        # get summary data
        self._fetch_precursor()

    def load_hull_data_3d(self,
                          intensity_min:int = 10,
                          ims_model:str = "gaussian",
                          mz_peak_width:float = 0.1,
                          averagine_prob_target:float = 0.95,
                          plot_feature:bool = False,
                          scan_range:int = 80) -> DataFrame:
        """Estimate convex hull of feature and return data points inside hull.

        Args:
            intensity_min (int, optional): Minimal peak intensity considered
              as signal. Defaults to 10.
            ims_model (str, optional): Model use in estimation of feature's
              width in IMS dimension. Defaults to "gaussian".
            mz_peak_width (float, optional): Expected width of a peak in mz
              dimension. Defaults to 0.1.
            averagine_prob_target (float, optional): Probability mass
              of averagine model's poisson distribution covered with
              extracted isotopic peaks . Defaults to 0.95.
            plot_feature (bool, optional): If true a scatterplot of
              feature is printed. Defaults to False.
            scan_range (int, optional): This parameter is handling
              the number of scans used to infer the scan boundaries
              of the monoisotopic peak. Defaults to 80.

        Returns:
            DataFrame: DataFrame with points in convex hull (scan,mz,intensity)
        """
        # bounds of monoisotopic peak based on arguments
        scan_min_init = int(self.scan_number//1)-scan_range//2
        scan_max_init = int(self.scan_number//1)+scan_range//2

         # via averagine calculate how many peaks should be considered.
        peak_n = self.get_num_peaks(self.monoisotopic_mz,
                                    self.charge,
                                    averagine_prob_target)
        mz_min_estimated = self.monoisotopic_mz-mz_peak_width
        mz_max_estimated = self.monoisotopic_mz+\
          (peak_n-1)*1/self.charge+mz_peak_width

        if ims_model in ["gaussian"]:
            # small mz window
            mz_min_init = self.monoisotopic_mz-mz_peak_width
            mz_max_init = self.monoisotopic_mz+mz_peak_width

            # extract monoisotopic peak
            frame_init = self.dataset_pointer.get_frame(self.frame_id)\
            .filter_ranged(scan_min_init,
                            scan_max_init,
                            mz_min_init,
                            mz_max_init,
                            intensity_min)
            # calculate profile of monoisotopic peak
            mono_profile_data = self.get_monoisotopic_profile(
                                          self.monoisotopic_mz,
                                          self.scan_number,
                                          frame_init,
                                          scan_range//2,
                                          mz_peak_width/2)
            # estimate scan boundaries
            scan_min_estimated,scan_max_estimated = self._get_scan_boundaries(
                                                              mono_profile_data,
                                                              ims_model)

        elif ims_model == "DBSCAN":
            dbscan_mz_window_factor = 5
            mz_min_init = self.monoisotopic_mz-dbscan_mz_window_factor*mz_peak_width
            mz_max_init = self.monoisotopic_mz+dbscan_mz_window_factor*mz_peak_width

            frame_init = self.dataset_pointer.get_frame(self.frame_id)\
                        .filter_ranged(
                            mz_min = mz_min_init,
                            mz_max = mz_max_init,
                            intensity_min = intensity_min)
            mzs_init = frame_init.mz()
            scans_init = frame_init.scan()
            rts_init = np.repeat(frame_init.frame_id(),mzs_init.shape[0])
            intensity_init = frame_init.intensity()
            points = np.vstack([rts_init,scans_init,mzs_init,intensity_init]).T
            # add monoisotopic peak
            # monoisotopic peak point as [rt (frame_id), scan, mz, arbitrary intensity = 100]
            mi_point = [self.frame_id,self.scan_number,self.monoisotopic_mz,100]
            # store at end of points via vstack and store position to get cluster id
            mi_position = len(points)
            points_with_mi = np.vstack([points,mi_point])
            # cluster data via DBSCAN
            clustered_data = cluster_precursors_dbscan(
                                                    points_with_mi,
                                                    epsilon=2,
                                                    min_samples=5)
            # get cluster id of added mono isotopic peak
            mi_cluster_id = clustered_data.label[mi_position]
            if plot_feature:
                plt.scatter(x=clustered_data.mz,y=clustered_data.scan,c=clustered_data.label,alpha=0.2)
                plt.scatter(x=clustered_data.mz[mi_position],y=clustered_data.scan[mi_position],color="black",marker="s")
                plt.show()
            if mi_cluster_id != -1:
                # cluster containing MI peak was found, select all points from it
                mi_selection = clustered_data.label == mi_cluster_id
                mi_cluster = clustered_data[mi_selection]
                # scan max and min are estimated to be highest and lowest
                # scan number in cluster, respectively
                scan_max_estimated = int(mi_cluster.scan.max())
                scan_min_estimated = int(mi_cluster.scan.min())
            else:
                raise ValueError("Monoisotopic peak cluster could not be found,\
                  use different method for estimation of scan width")

        # extract feature's hull data
        frame = self.dataset_pointer.get_frame(self.frame_id).filter_ranged(
                                                          scan_min_estimated,
                                                          scan_max_estimated,
                                                          mz_min_estimated,
                                                          mz_max_estimated,
                                                          intensity_min)
        scans = frame.scan()
        mzs = frame.mz()
        intensity = frame.intensity()

        # plot
        if plot_feature:
            scatter_3d = plt.figure()
            ax = scatter_3d.add_subplot(111,projection="3d")
            ax.scatter(mzs,scans,intensity)
        # return as DataFrame
        return DataFrame({"Scan":scans,"Mz":mzs,"Intensity":intensity},
                          dtype="float")

    def _fetch_precursor(self)-> None:
        """
        get row data from experiment's precursors table
        """
        raw_data = self.dataset_pointer
        feature_data_row = raw_data.get_selected_precursor_by_id(self.precursor_id)
        self.monoisotopic_mz = feature_data_row["MonoisotopicMz"].values[0]
        self.charge = feature_data_row["Charge"].values[0]
        self.scan_number = feature_data_row["ScanNumber"].values[0]
        self.frame_id = feature_data_row["Parent"].values[0]

    def _get_precursor_summary(self) -> None:
        """
        returns precursor row data
        """
        summary = DataFrame({"MonoisotopicMz":self.monoisotopic_mz,
                             "Charge":self.charge,
                             "ScanNumber":self.scan_number,
                             "FrameID":self.frame_id
                                })
        return summary

    def _get_scan_boundaries(self,
                             datapoints:np.ndarray,
                             ims_model:str="gaussian",
                             cut_off_left:float=0.01,
                             cut_off_right:float=0.99,
                             skip_zeros:bool=True) -> tuple:
        """Estimate minimum scan and maximum scan.

        Args:
            datapoints (np.ndarray): Scan, Intensity data from monoisotopic peak
              as 2D array: [[scan1,intensity_n],...,[scan_n,intensity_n]]
            ims_model (str, optional): Model of an IMS peak.
              Defaults to "gaussian".
            cut_off_left (float, optional): Probability mass to ignore on
              "left side". Defaults to 0.05.
            cut_off_right (float, optional): Probability mass to ignore on
              "right side". Defaults to 0.95.
            skip_zeros (bool, optional): Wether to ignore zero intensities in
              monoisotopic_profile. Defaults to True.

        Returns:
            tuple (int,int): (lower scan bound, upper scan bound)
        """
        # model functions to fit
        def _gauss(data,a,mu,sig):
            return a/(sig*np.sqrt(2*np.pi))*\
                   np.exp(-0.5*np.power((data-mu)/(sig),2))
        if skip_zeros:
            datapoints = datapoints[datapoints[:,1]!=0]
        # extract data
        x = datapoints.T[0]
        y = datapoints.T[1]

        if ims_model == "gaussian":
            # fit model function

            param_opt,param_cov = curve_fit(_gauss, # pylint: disable=unused-variable
                                            x,
                                            y,
                                            bounds=([y.min(),x.min(),0],
                                            [y.max(),x.max(),inf]))

            # instantiate a normal distribution with calculated parameters
            fit_dist = norm(param_opt[1],param_opt[2])
            # calculate lower and upper quantiles
            lower = fit_dist.ppf(cut_off_left)
            upper = fit_dist.ppf(cut_off_right)

            return(int(lower//1),int(upper//1+1))

        else:
            raise NotImplementedError("This model is not implemented")


    @staticmethod
    def get_num_peaks(monoisotopic_mz: float,
                      charge: int,
                      prob_mass_target: float = 0.95) -> int:
        """Calculation of number of isotopic peaks
        by averagine model.

        Args:
            monoisotopic_mz (float): Position of monoisotopic peak.
            charge (int): Charge of peptide
            prob_mass_target(float, optional): Minimum probability mass
              of poisson distribution, that shall be covered
              (beginning with monoisotopic peak). Defaults to 0.95.
        Returns:
            int: Number of relevant peaks
        """
        # calculate lam of averagine poisson distribution
        mass = monoisotopic_mz * charge
        lam = 0.000594 * mass - 0.03091
        poisson_averagine = poisson(lam)

        # find number of peaks necessary to cover for
        # given prob_mass_target
        prob_mass_covered = 0
        peak_number = 0
        while prob_mass_covered < prob_mass_target:
            # calculation of probability mass of a single peak
            peak_mass = poisson_averagine.pmf(peak_number)
            prob_mass_covered += peak_mass
            peak_number += 1
            # DEBUG
            # print(f"{peak_number} peak. Probability Mass : {peak_mass}")

        return peak_number

    @staticmethod
    def get_monoisotopic_profile(monoisotopic_mz:float,
                                scan_number:float,
                                frame_slice:TimsFrame,
                                scan_range:int = 20,
                                mz_range:float=0.05) -> np.ndarray:
        """Gets profile of monoisotopic peak in IMS dimension.

        Sums up peaks per scan that have a mz value close enough (mz_range)
        to monoisotopic peak mz.

        Args:
            monoisotopic_mz (float): Mz value of peak.
            scan_number (float): ScanNumber of peak.
            frame_slice (TimsFrame): Slice of monoisotopic peak
            scan_range (int, optional): Number of scans to consider.
              Defaults to 20.
            mz_range (float, optional): Maximal distance of a peak
              to monoisotopic mz to be considered in calculation.
              Defaults to 0.05.
        Returns:
            np.ndarray: 2D Array of structure [[scan,intensity],...]
        """
        # lowest scan number and highest scan number
        scan_l = int(scan_number//1)-scan_range//2
        scan_u = scan_l + scan_range
        considered_scans = np.arange(scan_l,scan_u)

        # extract values from passed TimsFrame slice of MI peak
        scans = frame_slice.scan().copy()
        mzs = frame_slice.mz().copy()
        intensities = frame_slice.intensity().copy()

        idxs = np.zeros((scan_range,2))

        for i,scan_i in enumerate(considered_scans):
            # only view points in mz range and current scan
            mask = (scans!=scan_i)|\
                   (mzs<monoisotopic_mz-mz_range)|\
                   (mzs>monoisotopic_mz+mz_range)
            intensities_ma = np.ma.MaskedArray(intensities,
                                               mask = mask)
            # sum these points up (intensities) and store in 2D array
            intensity_cumulated = np.ma.sum(intensities_ma) if\
                                  intensities_ma.count()>0 else\
                                  0
            idxs[i] = [scan_i,intensity_cumulated]


        return idxs

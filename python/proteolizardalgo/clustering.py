import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from proteolizardalgo.utility import peak_width_preserving_mz_transform
from hdbscan import HDBSCAN


def cluster_precursors_dbscan(precursor_points,
                              epsilon: float = 1.7,
                              min_samples: int = 7,
                              metric: str = 'euclidean',
                              cycle_scaling: float = -.4,
                              scan_scaling: float = .4,
                              resolution: int = 50_000):
    """
    cluster the precursors of a given (potentially filtered) precursor slice of timsTOF data with dbscan
    :param precursor_points: a Slice that must contain precursor data
    :param epsilon: DBSCAN epsilon
    :param min_samples: DBSCAN min_samples
    :param metric: DBSCAN metric
    :param cycle_scaling: a scale factor for rt, will be calculated as index / 2^cycle_scaling
    :param scan_scaling: a scale factor for ion mobility, will be calculated as index / 2^scan_scaling
    :param resolution: resolution of device in mz dimension
    :return: a pandas DataFrame containing precursor points with coordinates and labels from clustering
    """

    # get points from slice
    points = precursor_points

    # make copy to avoid return of scaled values
    rt_dim = precursor_points.frame.values
    f = np.sort(np.unique(rt_dim))
    f_idx = dict(np.c_[f, np.arange(f.shape[0])])
    rt_dim = [f_idx[x] for x in precursor_points.frame.values]

    dt_dim = precursor_points.scan.values
    mz_dim = precursor_points.mz.values

    # scale values according to parameters
    rt_dim_scaled = rt_dim / np.power(2, cycle_scaling)
    dt_dim_scaled = dt_dim / np.power(2, scan_scaling)
    mz_dim_scaled = peak_width_preserving_mz_transform(mz_dim, resolution=resolution)

    # cluster
    clusters = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=-1,
                      metric=metric).fit(np.vstack([rt_dim_scaled, dt_dim_scaled, mz_dim_scaled]).T)

    # return results as dataframe
    return pd.DataFrame(np.vstack([precursor_points.frame.values,
                                   precursor_points.scan.values,
                                   precursor_points.mz.values,
                                   precursor_points.intensity.values,
                                   clusters.labels_]).T,
                        columns=['cycle', 'scan', 'mz', 'intensity', 'label'])


def cluster_precursors_hdbscan(precursor_points,
                               algorithm: str = 'best',
                               alpha: float = 1.0,
                               approx_min_span_tree: bool = True,
                               gen_min_span_tree=True,
                               leaf_size=40,
                               min_cluster_size=7,
                               min_samples: int = 7,
                               p=None,
                               metric: str = 'euclidean',
                               cycle_scaling: float = -.4,
                               scan_scaling: float = .4,
                               resolution: int = 50_000,
                               mz_scaling: float = 0.0
                               ):
    """
    cluster the precursors of a given (potentially filtered) precursor slice of timsTOF data with dbscan
    :param mz_scaling:
    :param precursor_points:
    :param algorithm:
    :param alpha:
    :param approx_min_span_tree:
    :param gen_min_span_tree:
    :param leaf_size:
    :param min_cluster_size:
    :param p:
    :param experiment_slice: a Slice that must contain precursor data
    :param min_samples: DBSCAN min_samples
    :param metric: DBSCAN metric
    :param cycle_scaling: a scale factor for rt, will be calculated as index / 2^cycle_scaling
    :param scan_scaling: a scale factor for ion mobility, will be calculated as index / 2^scan_scaling
    :param resolution: resolution of device in mz dimension
    :return: a pandas DataFrame containing precursor points with coordinates and labels from clustering
    """

    # get points from slice
    points = precursor_points

    # make copy to avoid return of scaled values
    rt_dim = np.copy(points[:, 0])
    dt_dim = np.copy(points[:, 1])
    mz_dim = np.copy(points[:, 2])

    # scale values according to parameters
    rt_dim_scaled = rt_dim / np.power(2, cycle_scaling)
    dt_dim_scaled = dt_dim / np.power(2, scan_scaling)
    mz_dim_scaled = peak_width_preserving_mz_transform(mz_dim, resolution=resolution) / np.power(2, mz_scaling)

    hdbscan = HDBSCAN(algorithm=algorithm,
                      alpha=alpha,
                      approx_min_span_tree=approx_min_span_tree,
                      gen_min_span_tree=gen_min_span_tree,
                      leaf_size=leaf_size,
                      metric=metric,
                      min_cluster_size=min_cluster_size,
                      min_samples=min_samples,
                      p=p)

    # cluster
    clusters = hdbscan.fit(np.vstack([rt_dim_scaled, dt_dim_scaled, mz_dim_scaled]).T)

    # return results as dataframe
    return pd.DataFrame(np.vstack([points[:, 0], points[:, 1], points[:, 2], points[:, 3],
                                   clusters.labels_, clusters.probabilities_]).T,
                        columns=['cycle', 'scan', 'mz', 'intensity', 'label', 'probability'])


def calculate_cluster_statistics(cluster_table):
    # separate noise and signal
    clusters = cluster_table[cluster_table.label != -1]
    noise = cluster_table[cluster_table.label == -1]

    # create some on cluster statistics
    sum_intensity_clusters = clusters.groupby(['label'])['intensity'].sum().sum()
    sum_intensity_noise = noise.groupby(['label'])['intensity'].sum().sum()

    intensity_ratio = np.round(sum_intensity_clusters / sum_intensity_noise, 3)

    num_points_clusters = clusters.shape[0]
    num_points_noise = noise.shape[0]

    points_ratio = np.round(num_points_clusters / num_points_noise, 3)

    num_clusters = int(clusters.label.max())

    table_dict = {'Total': [num_points_clusters + num_points_noise,
                            sum_intensity_clusters + sum_intensity_noise, num_clusters],
                  'Cluster': [num_points_clusters, sum_intensity_clusters, num_clusters],
                  'Noise': [num_points_noise, sum_intensity_noise, 0],
                  'Ratio': [points_ratio, intensity_ratio, 0]}

    summary_table = pd.DataFrame(table_dict, index=['Points', 'Intensity', 'Clusters'])

    sum_table = clusters.groupby('label')

    return summary_table, sum_table

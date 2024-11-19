import math
import pickle
from abc import ABC, abstractmethod
from os import PathLike
from typing import AnyStr, Sequence, Any

import cv2
import mosaicking
from pathlib import Path
import numpy as np
from numpy import typing as npt
from scipy.spatial.transform import Slerp

import mosaicking.core.db
import mosaicking.core.interface
import mosaicking.transformations
from mosaicking import preprocessing, utils, registration

import networkx as nx

import re
import logging

import inspect

def copy_signature(base):
    def decorator(func):
        func.__signature__ = inspect.signature(base)
        return func
    return decorator

# Get a logger for this module
module_logger = logging.getLogger(__name__)


class Mapper:
    logger = logging.getLogger(__name__)

    def __init__(self, output_width: int, output_height: int, alpha_blend: float = 0.5):
        assert 0.0 <= alpha_blend <= 1.0, "Alpha blend must in interval [0, 1]."
        self._alpha = alpha_blend
        self._canvas, self._canvas_mask = self._create_canvas(output_width, output_height)
        self._flag = True

    def _create_canvas(self, output_width: int, output_height: int) -> tuple[npt.NDArray[np.uint8] | cv2.cuda.GpuMat,
                                                                       npt.NDArray[np.uint8] | cv2.cuda.GpuMat]:
        if mosaicking.HAS_CUDA:
            return self._cuda_create_canvas(output_width, output_height)
        return self._cpu_create_canvas(output_width, output_height)

    @staticmethod
    def _cuda_create_canvas(output_width: int, output_height: int) -> tuple[cv2.cuda.GpuMat,
                                                                      cv2.cuda.GpuMat]:
        tmp, tmp_mask = Mapper._cpu_create_canvas(output_width, output_height)
        output = cv2.cuda.GpuMat(tmp)
        output_mask = cv2.cuda.GpuMat(tmp_mask)
        return output, output_mask

    @staticmethod
    def _cpu_create_canvas(output_width: int, output_height: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_mask = np.zeros((output_height, output_width), dtype=np.uint8)
        return output, output_mask

    def _update_cuda(self, image: cv2.cuda.GpuMat, H: npt.NDArray[float], stream: cv2.cuda.Stream = None, mask: cv2.cuda.GpuMat = None):
        image = preprocessing.make_bgr(image)
        dsize = self._canvas_mask.size()
        width, height = image.size()
        if mask is None:
            mask = cv2.cuda.GpuMat(255 * np.ones((height, width), dtype=np.uint8))
        warped = cv2.cuda.warpPerspective(image, H, dsize, None, cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = cv2.cuda.warpPerspective(mask, H, dsize, None, cv2.INTER_CUBIC,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # Create binary masks for the regions
        output_mask_bin = cv2.cuda.threshold(self._canvas_mask, 1, 255, cv2.THRESH_BINARY)[1]
        warped_mask_bin = cv2.cuda.threshold(warped_mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Identify the intersecting and exclusive regions
        mask_intersect = cv2.cuda.bitwise_and(output_mask_bin, warped_mask_bin)
        warped_mask_only = cv2.cuda.bitwise_and(warped_mask_bin, cv2.cuda.bitwise_not(output_mask_bin))

        # Copy the warped region to the exclusively warped region (that's it for now)
        warped.copyTo(warped_mask_only, self._canvas)
        # Update the output mask with the warped region mask
        warped_mask_only.copyTo(warped_mask_only, self._canvas_mask)

        # Blend the intersecting regions
        # Prepare an alpha blending mask
        alpha_gpu = cv2.cuda.normalize(mask_intersect, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat())
        alpha_gpu = alpha_gpu.convertTo(alpha_gpu.type(), alpha=self._alpha)
        # Alpha blend the intersecting region
        blended = alpha_blend_cuda(self._canvas, warped, alpha_gpu)
        # Convert to 8UC3
        blended = cv2.cuda.merge(tuple(
            cv2.cuda.normalize(channel, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U, cv2.cuda.GpuMat()) for channel in
            cv2.cuda.split(blended)), cv2.cuda.GpuMat())
        blended.copyTo(mask_intersect, self._canvas)

        # cleanup
        mask.release()
        warped.release()
        warped_mask.release()
        output_mask_bin.release()
        warped_mask_bin.release()
        mask_intersect.release()
        warped_mask_only.release()
        alpha_gpu.release()
        blended.release()

    def _update_cpu(self, image: npt.NDArray[np.uint8], H: npt.NDArray[float], mask: npt.NDArray[np.uint8] = None):
        image = preprocessing.make_bgr(image)
        dsize = self._canvas.shape[1::-1]
        warped = cv2.warpPerspective(image, H, dsize, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if mask is None:
            warped_mask = np.where(warped.any(axis=2), 255, 0).astype(np.uint8)
        else:
            warped_mask = cv2.warpPerspective(mask, H, dsize, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # erode the mask a little
        warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_ERODE,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        warped_mask_bin = warped_mask.copy()
        output_mask_bin = cv2.threshold(self._canvas_mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Identify the intersecting and exclusive regions
        mask_intersect = cv2.bitwise_and(output_mask_bin, warped_mask_bin)
        warped_mask_only = cv2.bitwise_and(warped_mask_bin, cv2.bitwise_not(output_mask_bin))

        # Copy the warped region to the exclusively warped region (that's it for now)
        self._canvas = np.where(warped_mask_only[..., None], warped, self._canvas)

        # Blend the intersecting regions
        blended = cv2.addWeighted(self._canvas, self._alpha, warped, 1 - self._alpha, 0.0)
        # Replace intersecting region with blend
        self._canvas = np.where(mask_intersect[..., None], blended, self._canvas)

        # Update the output mask with the warped region mask
        self._canvas_mask = cv2.bitwise_or(self._canvas_mask, warped_mask)

    def _create_keypoint_mask(self, keypoints: npt.NDArray[float], image_dims: tuple[int, int]) -> npt.NDArray[
        np.uint8]:
        """
        Creates a binary mask from keypoints, with the region enclosed by the keypoints set to 1.

        :param keypoints: List of (x, y) coordinates defining the feature region.
        :param image_shape: Dimensions of the image (width, height).
        :return: Binary mask with the enclosed region set to 255, and the rest set to 0.
        """
        mask = np.zeros(image_dims[1::-1], dtype=np.uint8)

        # Compute the convex hull of the keypoints
        hull = cv2.convexHull(np.array(keypoints).astype(np.int32))

        # Fill the convex hull area with white (255) to define the mask
        mask = cv2.fillConvexPoly(mask, hull, 255)

        return mask

    def update(self, image: np.ndarray | cv2.cuda.GpuMat, H: np.ndarray, stream: cv2.cuda.Stream = None, keypoints: npt.NDArray[float] | Sequence[cv2.KeyPoint] = None):
        mask = None
        if keypoints and isinstance(keypoints[0], cv2.KeyPoint):
            keypoints = cv2.KeyPoint.convert(keypoints)
            width, height = image.size() if isinstance(image, cv2.cuda.GpuMat) else image.shape[1::-1]
            mask = self._create_keypoint_mask(keypoints, (width, height))
        if mosaicking.HAS_CUDA:
            if mask is not None and isinstance(mask, np.ndarray):
                mask = cv2.cuda.GpuMat(mask.copy())
            if isinstance(image, np.ndarray):
                image = cv2.cuda.GpuMat(image.copy())
            self._update_cuda(image, H, stream, mask)
        else:
            self._update_cpu(image, H, mask)

    def release(self):
        if mosaicking.HAS_CUDA:
            self._canvas.release()
            self._canvas_mask.release()

    @property
    def output(self) -> npt.NDArray[np.uint8]:
        if isinstance(self._canvas, np.ndarray):
            return self._canvas
        return self._canvas.download()


class Mosaic(ABC):
    logger = logging.getLogger(__name__)
    _load_instance = False

    def __new__(cls, *args, **kwargs):
        if cls._load_instance:
            cls._load_instance = False  # Reset after loading
            return super().__new__(cls)
        return super().__new__(cls)

    def __init__(self,
                 project_path: AnyStr | PathLike | Path,
                 data_path: AnyStr | PathLike | Path = None,
                 reader_params: dict[str, Any] = None,
                 preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]] = None,
                 feature_types: Sequence[str] = ("ORB",),
                 extractor_kwargs: dict[str, Any] = None,
                 bovw_clusters: int = 500,
                 bovw_batchsize: int = 1000,
                 nn_top_k: int = 5,
                 intrinsics: dict[str, np.ndarray] = None,
                 orientation_path: AnyStr | PathLike | Path = None,
                 orientation_time_offset: float = 0.0,
                 min_matches: int = 10,
                 homography_type: str | registration.HomographyEstimationType = "partial",
                 epsilon: float = 1e-4,
                 min_sequence_length: int = None,
                 alpha: float = 0.5,
                 keypoint_roi: bool = False,
                 overwrite: bool = False,
                 force_cpu: bool = False,
                 ):
        # Call loading routine
        if not self._load_instance and not overwrite and Path(project_path, "mosaic.pkl").exists():
            loaded_instance = self.load(project_path)
            # Copy loaded instance attributes to this instance
            self.__dict__.update(loaded_instance.__dict__)
            return  # Skip reinitializing if loading

        # Init overrides
        self._overwrite = overwrite  # Overwrite db entries
        self._force_cpu = force_cpu  # Force CPU use

        self._project_path = project_path
        self._data_path = data_path
        # Init data reader
        self._reader_params = reader_params
        self._reader_obj = self._create_reader_obj()
        # Init preprocessor
        self._preprocessing_params = preprocessing_params
        self._preprocessor_pipeline, self._preprocessor_args = self._create_preprocessor_obj(preprocessing_params)
        # Init representation
        self._model = mosaicking.core.interface.ImageGraph(db_dir=self._project_path)  # Graph + database backend
        # Init feature extraction
        self._feature_types = feature_types
        self._extractor_kwargs = extractor_kwargs
        if extractor_kwargs is None:
            extractor_kwargs = dict()
        self._feature_extractor = registration.CompositeDetector(self._feature_types,
                                                                 force_cpu=force_cpu,
                                                                 **extractor_kwargs)
        # Init global feature extraction
        self._bovw_clusters = bovw_clusters
        self._bovw_batchsize = bovw_batchsize
        self._bovw = dict()
        self._nn_top_k = nn_top_k
        self._nn = dict()
        # Init orientation
        self._intrinsics = intrinsics
        self._orientation_path = orientation_path
        self._orientation_time_offset = orientation_time_offset
        # Init registration
        self._min_matches = min_matches                     # Minimum number of matches for valid registration
        homography_type = registration.HomographyEstimationType(homography_type)
        self._homography_estimator = registration.HomographyEstimation(homography_type)
        self._epsilon = epsilon                             # Minimum perspective determinant for valid registration.
        self._matcher = registration.CompositeMatcher()     # For feature matching
        # Init blending
        self._min_sequence_length = min_sequence_length
        self._alpha = alpha
        self._use_keypoint_roi = keypoint_roi


    @classmethod
    def load(cls, project_path: AnyStr | PathLike | Path = None):
        cls._load_instance = True
        project_path = project_path.resolve(True)
        with open(project_path / 'mosaic.pkl', 'rb') as f:
            cls.logger.info(f"Restoring project from {project_path / 'mosaic.pkl'}")
            return pickle.load(f)

    def save(self):
        self.logger.info(f"Saving project at {self._project_path / 'mosaic.pkl'}")
        with open(self._project_path / 'mosaic.pkl', 'wb') as f:
            pickle.dump(self, f)

    @abstractmethod
    def _create_reader_obj(self) -> utils.DataReader:
        """
        A method to create the reader object from self._reader_params.
        """

    @staticmethod
    def _create_preprocessor_obj(preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]]) -> tuple[preprocessing.Pipeline, Sequence[dict[str, Any]]]:
        if preprocessing_params is None:
            return preprocessing.Pipeline(tuple()), tuple()
        obj_strings, init_args, args = zip(*preprocessing_params)
        objs = preprocessing.parse_preprocessor_strings(*obj_strings)
        pipeline = preprocessing.Pipeline([o(**arg) for o, arg in zip(objs, init_args)])
        return pipeline, args

    def _load_orientations(self) -> None | Slerp:
        if self._orientation_path is None or self._orientation_time_offset is None:
            return None
        orientation_path = Path(self._orientation_path)
        orientation_path.resolve(True)
        return utils.load_orientation_slerp(orientation_path, self._orientation_time_offset)

    def extract_features(self):
        self.logger.info(f"Beginning feature extraction with features: {self._feature_extractor.feature_type()}.")
        for ret, idx, image_name, image in self._reader_obj:
            # bad frames are skipped
            if not ret:
                self.logger.warning(f"Got bad image, skipping.")
                continue
            image = self._preprocessor_pipeline.apply(image)  # Apply preprocessing to image
            image = preprocessing.make_gray(image)  # Convert to grayscale for feature detection
            dimensions = image.size() if isinstance(image, cv2.cuda.GpuMat) else image.shape[1::-1]
            node = self._model.register_node(image_name,
                                             dimensions)  # TODO: retrieve image calibration & distortion here
            features = self._feature_extractor.detect(image)
            self._model.add_features(node, features)

    def global_features(self):
        # Initialize the bovw models
        if not self._bovw or self._overwrite:
            self._bovw.update(registration.create_bovw(self._model, self._bovw_clusters, self._bovw_batchsize))
        # Map nodes into vocabulary table
        for node, features in self._model.nodes(data="features", default=None):
            if features is None:
                continue
            global_descriptors = {}
            for feature_type, feature in features().items():
                hist = registration.create_bovw_histogram(feature['descriptors'], self._bovw[feature_type])
                global_descriptors[feature_type] = hist
            self._model.add_global_features(node, global_descriptors)
        # Initialize the kNN models
        if not self._nn or self._overwrite:
            self._nn = registration.create_nn(self._model, self._nn_top_k)

    def node_knn(self, query_node_idx: int) -> tuple[Sequence[mosaicking.core.interface.Node], npt.NDArray[float]]:
        """
        Look up the K nearest neighbours (not including self) to the query node.
        :param query_node_idx:
        :return:
        """
        query_node = list(self._model.nodes)[query_node_idx]
        query_features = self._model.nodes(data="global_features", default=None)[query_node]()
        distance_accumulator = dict()
        for feature_type, global_features in query_features.items():
            distances, indices = self._nn[feature_type].kneighbors(global_features[None, ...])
            for distance, idx in zip(distances[0][1:], indices[0][1:]):
                if idx not in distance_accumulator:
                    distance_accumulator.update({idx: distance})
                else:
                    distance_accumulator[idx] += distance
        sorted_neighbors = np.stack(sorted(distance_accumulator.keys(), key=lambda x: distance_accumulator[x]))
        sorted_distances = np.stack([distance_accumulator[n] for n in sorted_neighbors])
        nodes = tuple(self._model.nodes)
        sorted_neighbor_nodes = tuple(nodes[n] for n in sorted_neighbors)
        return sorted_neighbor_nodes, sorted_distances

    # These methods rely on a schema for generating relationships (e.g. sequential, brute-force, chunking)
    @abstractmethod
    def match_features(self):
        """
        A method for matching features in self._features to self._matches
        """
        ...

    @abstractmethod
    def registration(self):
        """
        A method for registering transformations from matches in self._matches to self._transforms
        """
        ...

    @abstractmethod
    def generate(self):
        """
        A method for generating the output mosaics.
        """
        ...


class SequentialMosaic(Mosaic):
    logger = logging.getLogger(__name__)
    frame_number_extractor = re.compile(r"\d+$")

    # TODO: Add a healing strategy to find nice homographies between subgraphs.
    # TODO: Post process image to convex hull around good features.

    # Private methods
    def _create_reader_obj(self) -> utils.VideoPlayer:
        player_params = {} if self._reader_params is None else self._reader_params
        if not self._force_cpu and mosaicking.HAS_CUDA and mosaicking.HAS_CODEC:
            return utils.CUDAVideoPlayer(self._data_path, **player_params)
        return utils.CPUVideoPlayer(self._data_path, **player_params)

    def _close_video_reader(self):
        self._reader_obj.release()
        self._reader_obj = None

    def _get_frame_number(self, node: mosaicking.core.interface.Node) -> int:
        return int(self.frame_number_extractor.search(node.name).group())

    def match_features(self):
        self.logger.info(f"Beginning BF feature matching.")
        # Sequential matching
        num_nodes = self._model.number_of_nodes()
        # Go through each node in order, looking for features with a successor node.
        for idx, (node_prev, node_data_prev) in enumerate(self._model.nodes(data=True)):
            # If last node reached in previous, break
            if idx == num_nodes - 1:
                break
            features_prev = node_data_prev.get("features", None)
            # if no features, skip
            if not features_prev:
                continue
            # Get the current node
            node = list(self._model.nodes)[idx + 1]  # minus 1
            features = self._model.nodes(data='features', default=None)[node]
            # if no features, skip
            if not features:
                continue
            # Retrieve the features from the db
            features = features()
            features_prev = features_prev()
            descriptors_prev = {feature_type: extract_features['descriptors'] for feature_type, extract_features in features_prev.items()}
            descriptors = {feature_type: extract_features['descriptors'] for feature_type, extract_features in features.items()}
            all_matches = self._matcher.knn_match(descriptors, descriptors_prev)
            good_matches = dict()
            # Apply Lowe's distance ratio test to acquire good matches.
            for feature_type, matches in all_matches.items():
                good_matches.update({feature_type: tuple(m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance)})
            num_matches = sum([len(m) for m in good_matches.values()])
            self.logger.debug(f"Good Matches {self._get_frame_number(node_prev)} - {self._get_frame_number(node)}: {num_matches}")
            # Perspective Homography requires at least 4 matches, if there aren't enough matches then don't create edge.
            if num_matches < self._min_matches:
                self.logger.debug(f"Not enough matches {self._get_frame_number(node_prev)} - {self._get_frame_number(node)}: {num_matches} < {self._min_matches}")
                continue
            self._model.register_edge(node_prev, node)
            self._model.add_matches(node_prev, node, good_matches)

    def registration(self):
        self.logger.info(f"Beginning local registration using {self._homography_estimator.method} homography estimator.")
        # Here, iterate through each subgraph and estimate homography.
        # If the homography quality is bad, then we prune the graph.
        to_filter = []
        for subgraph in (self._model.subgraph(c) for c in nx.weakly_connected_components(self._model)):
            subgraph.new_db(self._model.graph['db_dir'])  # TODO: this is rather clunky, but it works. Ideally subgraph would init the correct db when initializing the copy of ImageGraph.
            for node_prev, node, edge_data in subgraph.edges(data=True):
                features_prev = subgraph.nodes[node_prev]['features']()
                features = subgraph.nodes[node]['features']()
                matches = edge_data['matches']()
                # concatenate the lists of matched keypoints for each feature type
                kp_prev, kp = [], []
                for feature_type in features:
                    for match in matches[feature_type]:
                        kp_prev.append(features_prev[feature_type]['keypoints'][match.trainIdx])
                        kp.append(features[feature_type]['keypoints'][match.queryIdx])
                kp_prev = cv2.KeyPoint.convert(kp_prev)
                kp = cv2.KeyPoint.convert(kp)
                H, inliers = self._homography_estimator(kp, kp_prev, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                reproj_error = registration.compute_reprojection_error(H, kp, kp_prev)
                frac_inliers = inliers.sum() / inliers.size
                self.logger.debug(f"Registration result {self._get_frame_number(node_prev)} - {self._get_frame_number(node)} error {reproj_error:.2f}, inlier % {frac_inliers * 100:.2f}")
                # If H isn't good, then remove the edge.
                if not find_nice_homographies(H, self._epsilon) or H is None:
                    to_filter.append((node_prev, node))
                    continue
                # Otherwise
                self._model.add_registration(node_prev, node, H, reproj_error, frac_inliers)
        # Prune the graph for all bad homographies.
        self._model.remove_edges_from(to_filter)

    def _add_orientation(self):
        orientations = self._load_orientations()
        if orientations is None:
            self.logger.info("No orientation file specified. Skipping.")
            return
        K = self._intrinsics.get("K", None)
        if K is None:
            self.logger.warning("No intrinsic matrix provided. Skipping orientations.")
            return
        K_inv = mosaicking.transformations.inverse_K(K)
        for node in self._model.nodes():
            pt = self._orientation_time_offset + self._get_frame_number(node) / self._reader_obj.fps
            # TODO: have a 0.01 s slop here, make this public to user?
            if abs(pt - orientations.times.min()) < 10e-2:
                R = orientations.rotations[0]
            elif abs(pt - orientations.times.max()) < 10e-2:
                R = orientations.rotations[-1]
            elif orientations.times.min() <= pt <= orientations.times.max():
                R = orientations(pt)
            else:
                # TODO: pt can be outside of interpolant bounds, warning the user here but could cause trouble down the
                #  pipeline.
                self.logger.warning(f"playback time outside of interpolant bounds; not adding to Node.")
                continue
            self._model.nodes[node]['H0'] = K @ R.as_matrix() @ K_inv  # Apply 3D rotation as projection homography

    def _prune_unstable_graph(self, stability_threshold: float):
        # TODO: homography can be stable, but still extremely warped.
        #  Scan through corners to find outliers (i.e. where bad warps are likely).
        """
        Stabilize the graph by pruning unstable edges iteratively.

        Args:
            stability_threshold: The threshold for determining stability.

        Returns:
            stabilized_graph: The pruned and stabilized graph.
        """
        c = 0
        flag = True
        while flag:
            flag = False  # this flag needs to be flipped to keep pruning
            c = c + 1
            # Get all the subgraphs
            subgraphs = list(nx.weakly_connected_components(self._model))
            self.logger.info(f"Iteration: {c}, {len(subgraphs)} subgraphs.")
            for subgraph in (self._model.subgraph(c).copy() for c in subgraphs):
                if len(subgraph) < 2:
                    continue
                subgraph.new_db(self._model.graph["db_dir"])
                H = mosaicking.core.interface._propagate_homographies(subgraph)  # Get the homography sequence for the subgraph
                # Find the first unstable transformation
                is_good = find_nice_homographies(H, stability_threshold)

                # If all transformations are stable, continue on
                if all(is_good):
                    continue

                # Find the first unstable transformation's index
                unstable_index = is_good.tolist().index(False)

                # edge case: it's the first Node. Prune edge 0 -> 1
                unstable_index = unstable_index + 1 if unstable_index == 0 else unstable_index

                # Find the corresponding node in the subgraph
                nodes = list(nx.topological_sort(subgraph))

                unstable_node = nodes[unstable_index]

                # Get the predecessor of the unstable node
                predecessors = list(subgraph.predecessors(unstable_node))
                if not predecessors:
                    raise ValueError("No predecessors found, something went wrong with the graph structure.")

                # Prune the graph by removing the edge that leads to the unstable node
                pred_node = predecessors[0]
                self.logger.info(f"Pruning edge: {pred_node} -> {unstable_node}")
                self._model.remove_edge(pred_node, unstable_node)
                flag = True  # flag to search for pruning again

    def global_registration(self):
        self.logger.info("Beginning global registration.")
        self._add_orientation()  # Add in extrinsic rotations as homographies to valid Nodes.
        self._prune_unstable_graph(1e-4)  # Prune the bad absolute homographies
        # Assign the absolute homography to each node in each subgraph
        subgraphs = list(nx.weakly_connected_components(self._model))
        self.logger.info(f"Found {len(subgraphs)} subgraphs.")
        for subgraph in (self._model.subgraph(c).copy() for c in
                         subgraphs):
            H = mosaicking.core.interface._propagate_homographies(subgraph)
            image_dims = mosaicking.core.interface._get_graph_image_dimensions(subgraph)
            min_x, min_y, _, _ = get_mosaic_dimensions(H, image_dims[:, 0], image_dims[:, 1])
            H_t = mosaicking.transformations.homogeneous_translation(-min_x, -min_y)[None, ...]
            H = H_t @ H
            # order the edges by dependency
            sorted_nodes = list(nx. topological_sort(subgraph))
            for homography, node in zip(H, sorted_nodes):
                self._model.nodes[node]['H'] = homography

    def _create_tile_graph(self, tile_size: tuple[int, int]) -> nx.Graph:
        """
        Create a graph where each node represents a tile of the output mosaic.
        Each tile will consist of frames whose warped coordinates overlap with the tile.

        :param tile_size: A tuple representing the width and height of each tile.
        """
        self.logger.info("Creating tiles.")
        # Initialize variables
        tile_graph = nx.Graph()  # The graph where tiles will be nodes
        # Iterate over all stable subgraphs in the registration object
        for subgraph_index, subgraph in enumerate((self._model.subgraph(c).copy() for c in
                                                   nx.weakly_connected_components(self._model))):
            if self._min_sequence_length is not None and len(subgraph) < self._min_sequence_length:
                self.logger.debug(f"Skipping subgraph (too short: {len(subgraph)})")
                continue
            # Get the topological sort of the subgraph (a valid path of transformations)
            sorted_nodes = list(nx.topological_sort(subgraph))
            sorted_H = np.stack([subgraph.nodes[n]['H'] for n in sorted_nodes], axis=0)  # Mosaic-frame Homographies for each node
            image_dims = mosaicking.core.interface._get_graph_image_dimensions(subgraph)
            # Get the output mosaic dimensions
            mosaic_dims = get_mosaic_dimensions(sorted_H, image_dims[:, 0], image_dims[:, 1])

            # Calculate tile coordinates
            tile_x = range(0, math.ceil(mosaic_dims[2]), tile_size[0])
            tile_y = range(0, math.ceil(mosaic_dims[3]), tile_size[1])
            self.logger.info(f"Sequence {subgraph_index}: Allocating {len(subgraph)} images to {len(tile_x)*len(tile_y)} tiles.")
            for tile_x_idx, tx in enumerate(tile_x):
                for tile_y_idx, ty in enumerate(tile_y):
                    # Create the bounding box for the current tile
                    tile_crns = np.array([[tx, ty],
                                          [tx + tile_size[0], ty],
                                          [tx + tile_size[0], ty + tile_size[1]],
                                          [tx, ty + tile_size[1]]], dtype=int)

                    # Initialize a dict to store homographies that overlap with this tile
                    tile_frames = {}
                    for node, data in subgraph.nodes(data=True):
                        H = data['H']
                        width, height = node.dimensions
                        # Calculate the warped corners of the frame
                        frame_crns = get_corners(H, width, height)
                        # Check if the frame_bbox overlaps with the current tile_bbox
                        if registration.bbox_overlap(tile_crns, frame_crns.squeeze().astype(int)):
                            # If there's an overlap, transform the homography for this tile and add to graph
                            # Get the homography
                            H_tile = mosaicking.transformations.homogeneous_translation(-tx, -ty) @ H
                            frame_number = self._get_frame_number(node)
                            tile_frames.update({frame_number: H_tile})
                    if tile_frames:
                        tile_graph.add_node((subgraph_index, int(tile_x_idx), int(tile_y_idx)), frames=tile_frames)
        return tile_graph

    def _get_good_keypoints(self, node: mosaicking.core.interface.Node) -> Sequence[cv2.KeyPoint] | None:
        """
        Finds the inlier keypoints used in matching for a particular image node. Default behaviour is to use the matches
        with the predecessor node, except when the query node is the first node in a sequence where it uses the successor..
        :param query_node_idx:
        :return: tuple of cv2.KeyPoint inliers or None if no predecessors or successors.
        :rtype: Sequence[cv2.KeyPoint]
        """
        prev_node = next(self._model.predecessors(node), None)  # Check if the node is the first in the subsequence
        next_node = next(self._model.successors(node), None)  # Check if the node is the last in the subsequence
        features = self._model.nodes(data="features", default=None)[node]()  # Get the features for this node
        # Degenerate case where sequence is a single image.
        if prev_node is None and next_node is None:
            return None
        # If first in the sequence, load matches with the next
        elif prev_node is None:
            matches = self._model.edges[node, next_node]['matches']()
            good_features = []
            # training idx = destination
            for feature_type, match_set in matches.items():
                for match in match_set:
                    good_features.append(features[feature_type]['keypoints'][match.trainIdx])
            return good_features
        # If last in the sequence, load matches with the previous
        elif next_node is None:
            matches = self._model.edges[prev_node, node]['matches']()
            good_features = []
            # training idx = destination
            for feature_type, match_set in matches.items():
                for match in match_set:
                    good_features.append(features[feature_type]['keypoints'][match.queryIdx])
            return good_features
        # If in middle of sequence somewhere, load matches from previous node
        matches = self._model.edges[prev_node, node]['matches']()
        good_features = []
        # training idx = destination
        for feature_type, match_set in matches.items():
            for match in match_set:
                good_features.append(features[feature_type]['keypoints'][match.queryIdx])
        return good_features

    def generate(self, tile_size: tuple[int, int] = None):
        tiles = self._create_tile_graph(tile_size)  # restructure the graph into tiles

        # Now iterate through the tiles
        for (subgraph_index, tile_x, tile_y), sequence in tiles.nodes(data="frames"):
            self.logger.info(f"Generating sequence {subgraph_index}: tile {tile_x}x{tile_y}")
            #  subgraph_index: which sequence this belongs to
            #  tile_x: top left corner of the tile in mosaic coordinates
            #  tile_y: top left corner of the tile in mosaic coordinates
            #  sequence: a dictionary of frame number and homography to apply to map image to the tile
            frame_numbers = list(sequence.keys())
            # Get frame range
            frame_min, frame_max = min(frame_numbers), max(frame_numbers)
            self._reader_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_min)
            # Construct mapper object
            mapper = Mapper(*tile_size, alpha_blend=self._alpha)
            for ret, pos, name, frame in self._reader_obj:
                if pos not in sequence:
                    continue
                frame = self._preprocessor_pipeline.apply(frame)
                keypoints = self._get_good_keypoints([n for n in self._model.nodes if n.name == name][0]) if self._use_keypoint_roi else None
                mapper.update(frame, sequence[pos], None, keypoints)
                if isinstance(frame, cv2.cuda.GpuMat):
                    frame.release()
            output_path = self._project_path.joinpath(f"seq_{subgraph_index}_tile_{tile_x}_{tile_y}.png")
            cv2.imwrite(str(output_path), mapper.output)
            mapper.release()

def find_nice_homographies(H: npt.NDArray[float], eps: float = 1e-3) -> npt.NDArray[bool]:
    """
    Homographies that have negative determinant will flip the coordinates.
    Homographies that have close to zero will also have undesirable behaviour (thin lines, extreme warping etc.).
    This function tries to identify these bad homographies.
    :param H: Homography matrix (3x3)
    :param eps: lower threshold that defines what is 'close to zero'
    :return:
    """
    if H.ndim > 2:
        det = np.linalg.det(H[:, :2, :2])
    else:
        det = np.linalg.det(H)
    return det > eps

def get_mosaic_dimensions(H: npt.NDArray[float], width: int | Sequence[int], height: int | Sequence[int]) -> Sequence[int]:
    """
    Given a transformation homography, and the width and height of the input image. Calculate the bounding box of the warped image.
    :param H: homography matrix (3x3) or (Nx3x3)
    :param width: width of image or tuple of widths (length N)
    :param height: height of image or tuple of heights (length N)
    :return: bounding box of the output mosaic with respect to the source frame of the first homography in H.
    :rtype: Sequence[int]
    """
    # Get the image corners
    dst_crn = get_corners(H, width, height)
    # Compute the top left and bottom right corners of bounding box
    # boundingRect uses integer rounding, so might not be appropriate for all uses.
    # NOTE: cv2.boundingRect suffers from overflow for large corners
    #return cv2.boundingRect(dst_crn.reshape(-1, 2).astype(np.float32))
    min_x, min_y = dst_crn.reshape(-1, 2).min(axis=0)
    max_x, max_y = dst_crn.reshape(-1, 2).max(axis=0)
    return min_x, min_y, max_x - min_x, max_y - min_y

def get_corners(H: npt.NDArray[float], width: int | Sequence[int], height: int | Sequence[int]) -> npt.NDArray[float]:
    """
    Given a transformation homography, and the width and height of the input image. Calculate the warped positions of
    the corners of the image.
    :param H: homography matrix (3x3) or (Mx3x3) for M homographies.
    :param width: width of image or tuple of widths (length M)
    :param height: height of image or tuple of heights (length M)
    :return: Warped corners of the images as array with shape Mx(M*3+1)x1x2.
    :rtype: npt.NDArray[float]
    """
    # Get the image corners as 4x1x2
    if isinstance(width, int):
        src_crn = np.array([[[0, 0]],
                            [[width - 1, 0]],
                            [[width - 1, height - 1, ]],
                            [[0, height - 1]]], np.float32) + 0.5
    # Multiple images: (M*3+1)x1x2
    elif len(width):
        crns = [[[0, 0]]]
        for w, h in zip(width, height):
            crns.extend([[[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
        src_crn = np.array(crns, np.float32) + 0.5
    else:
        raise ValueError("width and height must either be ints or sequences of ints with same length.")
    # If multiple homographies: stack src_crn to be Mx(M*3+1)x1x2
    if H.ndim > 2:
        src_crn = np.stack([src_crn] * len(H), 0)
    # If single homography: expand src_crn and H to be 1x4x1x2 and 1x3x3 respectively
    elif H.ndim == 2:
        src_crn = src_crn[None, ...]
        H = H[None, ...]
    # M homographies, N corners
    M, N = src_crn.shape[:2]
    # Make homogeneous (MxNx1x3)
    src_crn_h = np.concatenate((src_crn,
                                np.ones((M, N, 1, 1))), axis=-1)
    # Applying homographies using broadcasting
    dst_crn_h = np.swapaxes(H @ np.swapaxes(src_crn_h.squeeze(axis=2), 1, 2), 1, 2)
    # Convert to non-homogeneous form
    dst_crn = dst_crn_h[:, :, :2] / dst_crn_h[:, :, -1:]
    # Expand additional dimension to make MxNx1x2 form
    return dst_crn[:, :, None, :]

def alpha_blend_cuda(img1: cv2.cuda.GpuMat, img2: cv2.cuda.GpuMat, alpha_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """
    Perform alpha blending composition between two CUDA GpuMat images.
    :param img1: First image (GpuMat)
    :param img2: Second image (GpuMat)
    :param alpha_gpu: GpuMat of floats in range [0.0, 1.0], specifying the blending weight to be applied to img1. The
    complement weighting (1.0 - alpha_gpu) will be applied to img2.
    :return: alpha blended image (GpuMat)
    :rtype: cv2.cuda.GpuMat
    """
    one_gpu = cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type(), 1)
    alpha_inv_gpu = cv2.cuda.subtract(one_gpu, alpha_gpu)

    if img1.type() != alpha_gpu.type():
        img1 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img1)) + (alpha_gpu,), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    if img2.type() != alpha_gpu.type():
        img2 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img2)) + (alpha_inv_gpu, ), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    blended = cv2.cuda.alphaComp(img1, img2, cv2.cuda.ALPHA_OVER)

    return cv2.cuda.cvtColor(blended, cv2.COLOR_BGRA2BGR)


def main():
    logging.basicConfig(level=logging.INFO)
    args = utils.parse_args()
    start = args.start_time_secs or args.start_playtime or args.start_frame
    finish = args.finish_time_secs or args.finish_playtime or args.finish_frame
    frameskip = args.frame_skip
    reader_params = None
    mosaicking.HAS_CUDA = mosaicking.HAS_CUDA and args.force_cuda_off
    mosaicking.HAS_CODEC = mosaicking.HAS_CODEC and args.force_cudacodec_off
    if start or finish or frameskip:
        reader_params = {}
        if start:
            reader_params.update({"start": start})
        if finish:
            reader_params.update({"finish": finish})
        if frameskip:
            reader_params.update({"frame_skip": frameskip})
    calibration = None
    if args.calibration:
        calibration = utils.parse_intrinsics(args.calibration)
    mos = SequentialMosaic(project_path=args.project,
                           data_path=args.video,
                           reader_params=reader_params,
                           feature_types=args.feature_types,
                           bovw_clusters=args.bovw_clusters,
                           bovw_batchsize=args.bovw_batchsize,
                           nn_top_k=args.nn_top_k,
                           intrinsics=calibration,
                           orientation_path=args.orientation_path,
                           orientation_time_offset=args.orientation_time_offset,
                           min_matches=args.min_matches,
                           homography_type=args.homography_type,
                           epsilon=args.epsilon,
                           min_sequence_length=args.min_sequence_length,
                           alpha=args.alpha,
                           keypoint_roi=args.keypoint_roi,
                           overwrite=args.overwrite,
                           force_cpu=args.force_cpu
                           )
    if args.overwrite or not args.project.join("mosaic.pkl").exists() or not args.project.join("mosaicking.db").exists():
        mos.extract_features()
        mos.match_features()
        mos.registration()
        mos.global_registration()
    mos.generate((args.tile_size, args.tile_size))
    mos.save()


if __name__ == '__main__':
    main()

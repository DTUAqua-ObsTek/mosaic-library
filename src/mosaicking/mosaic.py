import itertools
import pickle
from abc import ABC, abstractmethod
from os import PathLike
from typing import Union, AnyStr, Sequence, Any

import cv2
import mosaicking
from pathlib import Path
import numpy as np
from numpy import typing as npt
from scipy.spatial.transform import Slerp

import mosaicking.core.db
import mosaicking.core.interface
import mosaicking.transformations
from mosaicking import preprocessing, utils, registration, core

import networkx as nx


import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


class Mapper:
    def __init__(self, output_width: int, output_height: int, alpha_blend: float = 0.5):
        assert 0.0 <= alpha_blend <= 1.0, "Alpha blend must in interval [0, 1]."
        self._alpha = alpha_blend
        self._canvas, self._canvas_mask = self._create_canvas(output_width, output_height)

    def _create_canvas(self, output_width: int, output_height: int) -> tuple[Union[np.ndarray, cv2.cuda.GpuMat],
                                                                       Union[np.ndarray, cv2.cuda.GpuMat]]:
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
    def _cpu_create_canvas(output_width: int, output_height: int) -> tuple[np.ndarray, np.ndarray]:
        output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_mask = np.zeros((output_height, output_width), dtype=np.uint8)
        return output, output_mask

    def _update_cuda(self, image: cv2.cuda.GpuMat, H: np.ndarray, stream: cv2.cuda.Stream = None):
        image = preprocessing.make_bgr(image)
        dsize = self._canvas_mask.size()
        width, height = image.size()
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

    def _update_cpu(self, image: np.ndarray, H: np.ndarray):
        image = preprocessing.make_bgr(image)
        dsize = self._canvas.shape[1::-1]
        height, width = image.shape[:2]
        warped = cv2.warpPerspective(image, H, dsize, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = np.where(warped.any(axis=2), 255, 0).astype(np.uint8)
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

    def update(self, image: Union[np.ndarray, cv2.cuda.GpuMat], H: np.ndarray, stream: cv2.cuda.Stream = None):
        if mosaicking.HAS_CUDA:
            if isinstance(image, np.ndarray):
                image = cv2.cuda.GpuMat(image.copy())
            self._update_cuda(image, H, stream)
        else:
            self._update_cpu(image, H)

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

    def __init__(self, data_path: Union[AnyStr, PathLike, Path] = None,
                 output_path: Union[AnyStr, PathLike, Path] = None,
                 feature_types: Sequence[str] = ('ORB',),
                 extractor_kwargs: dict[str, Any] = None,
                 preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]] = None,
                 intrinsics: dict[str, np.ndarray] = None,
                 orientation_path: Union[AnyStr, PathLike, Path] = None,
                 time_offset: float = 0.0,
                 verbose: bool = False,
                 caching: bool = True,
                 overwrite: bool = True,
                 force_cpu: bool = False,
                 player_params: dict[str, Any] = None,
                 min_matches: int = 10,
                 epsilon: float = 1e-4,
                 ):
        assert data_path is not None or output_path is not None, "data_path and output_path cannot both be unspecified."
        assert not Path(output_path).is_file(), "output_path is not a directory."

        if output_path is not None:
            output_path = Path(output_path)  # convert output_path to pathlib.Path
        if data_path is not None:
            data_path = Path(data_path).resolve(True)  # convert data_path to pathlib.Path (must exist)
        if orientation_path is not None:
            orientation_path = Path(orientation_path).resolve(True)  # convert orientation_path to pathlib.Path (must exist)

        self._meta = None  # initialize metadata variable
        # output_path specified, either an old configuration to be rerun or a new output.
        if output_path is not None:
            # load meta if meta.json exists, then it is a rerun.
            if output_path.joinpath("meta.pkl").exists() and not overwrite:
                with output_path.joinpath("meta.pkl", "rb").open() as f:
                    self._meta = pickle.load(f)
                # check data_path is correct
                assert "data_path" in self._meta, "meta missing attribute data_path."
                assert self._meta["data_path"] is not None, "meta attribute data_path undefined."
                if data_path is not None:
                    # Warn that data_path doesn't match meta.data_path
                    if self._meta["data_path"] != data_path:
                        logger.warning(f"meta.data_path does not match data_path argument, using meta.data_path. "
                                      f"Call with overwrite argument to overwrite meta.data_path.")
            # if meta.json doesn't exist, make sure data_path does.
            else:
                assert data_path is not None, "data_path undefined and meta.pkl doesn't exist."
                self._meta = dict(data_path=data_path,
                                  output_path=output_path,
                                  feature_types=feature_types,
                                  preprocessing_params=preprocessing_params,
                                  intrinsics=intrinsics,
                                  orientation_path=orientation_path,
                                  time_offset=time_offset,
                                  force_cpu=force_cpu,
                                  player_params=player_params,
                                  min_matches=min_matches,
                                  epsilon=epsilon)
        else:
            self._meta = dict(data_path=data_path,
                              output_path=data_path.with_name(data_path.stem + "_mosaic"),
                              feature_types=feature_types,
                              preprocessing_params=preprocessing_params,
                              intrinsics=intrinsics,
                              orientation_path=orientation_path,
                              time_offset=time_offset,
                              force_cpu=force_cpu,
                              player_params=player_params,
                              min_matches=min_matches,
                              epsilon=epsilon)

        self._verbose = verbose                                                     # For logging verbosity.
        self._reader_obj = self._create_reader_obj()                                # For reading the data.
        self._registration_obj = mosaicking.core.interface.ImageGraph(db_dir=output_path)  # For registration tracking.
        # For feature extraction
        if extractor_kwargs is None:
            extractor_kwargs = dict()
        self._feature_extractor = registration.CompositeDetector(self._meta["feature_types"], force_cpu=self._meta["force_cpu"], **extractor_kwargs)
        self._matcher = registration.CompositeMatcher()  # For feature matching
        self._preprocessor_pipeline, self._preprocessor_args = self._create_preprocessor_obj(preprocessing_params)
        self._caching = caching                          # Flag to cache feature extraction
        self._overwrite = overwrite                      # Flag to overwrite anything in output path

    def save(self):
        self._meta["output_path"].mkdir(parents=True, exist_ok=True)
        if not self._meta["output_path"].joinpath("meta.pkl").exists() or self._overwrite:
            if self._overwrite:
                logger.warning(f"Overwriting {self._meta['output_path'] / 'meta.pkl'}")
            with open(self._meta["output_path"] / "meta.pkl", "wb") as f:
                pickle.dump(self._meta, f)

    @classmethod
    def load(cls, meta_file: Path):
        meta_file.resolve(True)
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        return cls(**meta)


    @abstractmethod
    def _create_reader_obj(self) -> utils.DataReader:
        """
        A method to create the reader object from self._meta.
        """

    @staticmethod
    def _create_preprocessor_obj(preprocessing_params: Sequence[tuple[str, dict[str, Any], dict[str, Any]]]) -> tuple[preprocessing.Pipeline, Sequence[dict[str, Any]]]:
        if preprocessing_params is None:
            return preprocessing.Pipeline(tuple()), tuple()
        obj_strings, init_args, args = zip(*preprocessing_params)
        objs = preprocessing.parse_preprocessor_strings(*obj_strings)
        pipeline = preprocessing.Pipeline([o(**arg) for o, arg in zip(objs, init_args)])
        return pipeline, args

    def _load_features(self):
        cache_path = self._meta["output_path"].joinpath("features.pkl")
        assert cache_path.exists(), f"Features cache {cache_path} does not exist."
        with cache_path.open("rb") as f:
            features = pickle.load(f)
            self._features = tuple(utils.convert_feature_keypoints(feature) for feature in features)

    def _load_orientations(self) -> Union[None, Slerp]:
        orientations_path = self._meta["orientation_path"]
        time_offset = self._meta["time_offset"]
        if orientations_path is None or time_offset is None:
            return None
        assert orientations_path.exists(), f"Orientation path {orientations_path} does not exist."
        return utils.load_orientation_slerp(orientations_path, time_offset)

    @abstractmethod
    def extract_features(self):
        """
        A method for extracting features to self._features
        """
        ...

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
    # TODO: Add a healing strategy to find nice homographies between subgraphs.

    # Private methods
    def _create_reader_obj(self) -> utils.VideoPlayer:
        player_params = {} if self._meta["player_params"] is None else self._meta["player_params"]
        if mosaicking.HAS_CUDA and mosaicking.HAS_CODEC:
            return utils.CUDAVideoPlayer(self._meta["data_path"], **player_params)
        return utils.CPUVideoPlayer(self._meta["data_path"], **player_params)

    def _close_video_reader(self):
        self._reader_obj.release()
        self._reader_obj = None

    # Required Overloads
    def extract_features(self):
        logger.info(f"Beginning feature extraction with features: {self._feature_extractor.feature_type()}.")
        for frame_no, (ret, frame) in enumerate(self._reader_obj):
            frame_name = self._reader_obj.next_name
            logger.debug(repr(self._reader_obj))
            # don't register a bad frame
            if not ret:
                continue
            frame = self._preprocessor_pipeline.apply(frame)  # Apply preprocessing to image
            frame = preprocessing.make_gray(frame)  # Convert to grayscale
            dimensions = frame.size() if isinstance(frame, cv2.cuda.GpuMat) else frame.shape[1::-1]
            features = self._feature_extractor.detect(frame)
            node = self._registration_obj.register_node(frame_name, dimensions)  # TODO: retrieve image calibration & distortion here
            self._registration_obj.add_features(node, features)

    def match_features(self):
        logger.info(f"Beginning BF feature matching.")
        # Sequential matching
        num_nodes = self._registration_obj.number_of_nodes()
        # Go through each node in order, looking for features
        for node_prev, node_data_prev in self._registration_obj.nodes(data=True):
            features_prev = node_data_prev.get("features", None)
            # if no features, skip
            if not features_prev:
                continue
            # get the next node
            node_id = node_prev.identifier + 1
            # If last node reached in previous, break
            if node_id == num_nodes:
                break
            node = list(self._registration_obj.nodes)[node_id]
            features = self._registration_obj.nodes(data='features', default=None)[node]
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
            logger.debug(f"Good Matches {node_id - 1} - {node_id}: {num_matches}")
            # Perspective Homography requires at least 4 matches, if there aren't enough matches then don't create edge.
            if num_matches < self._meta["min_matches"]:
                logger.debug(f"Not enough matches {node_id - 1} - {node_id}: {num_matches} < {self._meta['min_matches']}")
                continue
            self._registration_obj.register_edge(node_prev, node)
            self._registration_obj.add_matches(node_prev, node, good_matches)

    def registration(self):
        logger.info(f"Beginning local registration.")
        # Here, iterate through each subgraph and estimate homography.
        # If the homography quality is bad, then we prune the graph.
        to_filter = []
        for subgraph in (self._registration_obj.subgraph(c) for c in nx.weakly_connected_components(self._registration_obj)):
            subgraph.new_db(self._registration_obj.graph['db_dir'])  # TODO: this is rather clunky, but it works. Ideally subgraph would init the correct db when initializing the copy of ImageGraph.
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
                #H, inliers = cv2.findHomography(kp, kp_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                H, inliers = cv2.estimateAffine2D(kp, kp_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                H = np.concatenate((H, np.array([[0, 0, 1]], dtype=float)), axis=0)
                # If H is no good, then remove the edge.
                if not find_nice_homographies(H, self._meta["epsilon"]) or H is None:
                    to_filter.append((node_prev, node))
                    continue
                # Otherwise
                self._registration_obj.add_registration(node_prev, node, H)
        # Prune the graph for all bad homographies.
        self._registration_obj.remove_edges_from(to_filter)

    @staticmethod
    def _propagate_homographies(G: nx.DiGraph) -> np.ndarray:
        # Given a directed graph of homographies, return them as a sequence of homographies for each node.
        if len(G) == 1:
            return G[list(G.nodes)[0]].get("H0", np.eye(3)[None, ...])
        elif len(G) < 1:
            raise ValueError("No nodes in graph.")
        else:
            # order the edges by dependency
            sorted_edges = list(nx.topological_sort(nx.line_graph(G)))
            sorted_H = [G[u][v]['registration']() for u, v in sorted_edges]
            N0 = sorted_edges[0][0]  # first node in sorted graph
            # if node has an initial rotation
            H0 = G[N0].get("H0", np.eye(3))
            sorted_H = [H0] + sorted_H
            sorted_H = np.stack(sorted_H, axis=0)
            return np.array(tuple(
                itertools.accumulate(sorted_H, np.matmul)))  # Propagate the homographys to reference to first node.

    def _add_orientation(self):
        orientations = self._load_orientations()
        K = self._meta["intrinsics"]["K"]
        K_inv = mosaicking.transformations.inverse_K(K)
        if orientations is None:
            return
        for node in self._registration_obj.nodes():
            pt = self._meta['time_offset'] + node.identifier / self._reader_obj.fps
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
                logger.warning(f"playback time outside of interpolant bounds; not adding to Node.")
                continue
            self._registration_obj.nodes[node]['H0'] = K @ R.as_matrix() @ K_inv  # Apply 3D rotation as projection homography

    def _prune_unstable_graph(self, stability_threshold: float):
        # TODO: homography can be stable, but extremely warped.
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
            subgraphs = list(nx.weakly_connected_components(self._registration_obj))
            logger.info(f"Iteration: {c}, {len(subgraphs)} subgraphs.")
            for subgraph in (self._registration_obj.subgraph(c).copy() for c in subgraphs):
                if len(subgraph) < 2:
                    continue
                subgraph.new_db(self._registration_obj.graph["db_dir"])
                H = SequentialMosaic._propagate_homographies(subgraph)  # Get the homography sequence for the subgraph
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
                logger.info(f"Pruning edge: {pred_node} -> {unstable_node}")
                self._registration_obj.remove_edge(pred_node, unstable_node)
                flag = True  # flag to search for pruning again

    @staticmethod
    def _get_graph_image_dimensions(G: nx.DiGraph) -> np.ndarray:
        """Accumulate the 'dimensions' attribute of nodes into a Nx2 NumPy array."""
        if len(G) < 1:
            raise ValueError("No nodes in graph.")
        return np.array([node.dimensions for node in G.nodes()])


    def global_registration(self):
        logger.info("Beginning global registration.")
        self._add_orientation()  # Add in extrinsic rotations as homographies to valid Nodes.
        self._prune_unstable_graph(1e-4)  # Prune the bad absolute homographies
        # Assign the absolute homography to each node in each subgraph
        for subgraph in (self._registration_obj.subgraph(c).copy() for c in
                         nx.weakly_connected_components(self._registration_obj)):
            H = self._propagate_homographies(subgraph)
            image_dims = self._get_graph_image_dimensions(subgraph)
            min_x, min_y, _, _ = get_mosaic_dimensions(H, image_dims[:, 0], image_dims[:, 1])
            H_t = mosaicking.transformations.homogeneous_translation(-min_x, -min_y)[None, ...]
            H = H_t @ H
            # order the edges by dependency
            sorted_nodes = list(nx. topological_sort(subgraph))
            for homography, node in zip(H, sorted_nodes):
                self._registration_obj.nodes[node]['H'] = homography

    @staticmethod
    def _bbox_overlap(shape_1: npt.NDArray[int], shape_2: npt.NDArray[int]) -> bool:
        rect_1 = cv2.minAreaRect(shape_1)
        rect_2 = cv2.minAreaRect(shape_2)
        intersection_type, _ = cv2.rotatedRectangleIntersection(rect_1, rect_2)
        return intersection_type > cv2.INTERSECT_NONE

    def _create_tile_graph(self, tile_size: tuple[int, int]) -> nx.Graph:
        # iterate through every subgraph sequence of _registration_obj
        # get the topological sort of subgraph (a valid path)
        # get the sequence of transformations to apply to each node (include the first H0 transformation from N0).
        # Get output mosaic dimensions
        # Generate tile coordinates based on tile size parameter and output mosaic dimensions
        # For each tile
            # Create data structure for tile
            # Generate mosaic -> tile translation homography based on top left tile coordinates
            # For each node
                # warp corners to output mosaic coordinates using H0
                # determine if warped bbox overlaps with current tile coordinates
                # if overlap:
                    # H_tile = transform H0 of node with tile translation homography
                    # Add node to tile datastructure with H_tile
        """
            Create a graph where each node represents a tile of the output mosaic.
            Each tile will consist of frames whose warped coordinates overlap with the tile.

            :param tile_size: A tuple representing the width and height of each tile.
            """
        # Initialize variables
        tile_graph = nx.Graph()  # The graph where tiles will be nodes
        # Iterate over all stable subgraphs in the registration object
        for subgraph_index, subgraph in enumerate((self._registration_obj.subgraph(c).copy() for c in
                         nx.weakly_connected_components(self._registration_obj))):
            # Get the topological sort of the subgraph (a valid path of transformations)
            sorted_nodes = list(nx.topological_sort(subgraph))
            sorted_H = np.stack([subgraph.nodes[n]['H'] for n in sorted_nodes], axis=0)  # Mosaic-frame Homographies for each node
            image_dims = self._get_graph_image_dimensions(subgraph)
            # Get the output mosaic dimensions
            mosaic_dims = get_mosaic_dimensions(sorted_H, image_dims[:, 0], image_dims[:, 1])

            # Calculate tile coordinates
            tile_x = np.arange(0, mosaic_dims[2], tile_size[0])
            tile_y = np.arange(0, mosaic_dims[3], tile_size[1])

            for tx in tile_x:
                for ty in tile_y:
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
                        if self._bbox_overlap(tile_crns, frame_crns.squeeze().astype(int)):
                            # If there's an overlap, transform the homography for this tile and add to graph
                            # Get the homography
                            H_tile = mosaicking.transformations.homogeneous_translation(-tx, -ty) @ H
                            tile_frames.update({node.identifier: H_tile})
                    if tile_frames:
                        tile_graph.add_node((subgraph_index, tx, ty), frames=tile_frames)
        return tile_graph

    def generate(self, tile_size: tuple[int, int] = None, alpha: float = 1.0):
        logger.info("Creating tiles.")
        tiles = self._create_tile_graph(tile_size)  # restructure the graph into tiles

        # Now iterate through the tiles
        for (subgraph_index, tile_x, tile_y), sequence in tiles.nodes(data='frames'):
            logger.info(f"Generating sequence {subgraph_index}: tile {tile_x}x{tile_y}")
            #  subgraph_index: which sequence this belongs to
            #  tile_x: top left corner of the tile in mosaic coordinates
            #  tile_y: top left corner of the tile in mosaic coordinates
            #  sequence: a dictionary of frame number and homography to apply to map image to the tile
            frame_numbers = list(sequence.keys())
            # Get frame range
            frame_min, frame_max = min(frame_numbers), max(frame_numbers)
            self._reader_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_min)
            # Construct mapper object
            mapper = Mapper(*tile_size, alpha_blend=alpha)
            for frame_no, (ret, frame) in enumerate(self._reader_obj, start=frame_min):
                if frame_no not in sequence:
                    continue
                frame = self._preprocessor_pipeline.apply(frame)
                mapper.update(frame, sequence[frame_no])
                if isinstance(frame, cv2.cuda.GpuMat):
                    frame.release()
            output_path = self._meta["output_path"].joinpath(f"seq_{subgraph_index}_tile_{tile_x}_{tile_y}.png")
            cv2.imwrite(str(output_path), mapper.output)
            mapper.release()



def find_nice_homographies(H: np.ndarray, eps: float = 1e-3) -> npt.NDArray[bool]:
    """
    Homographies that have negative determinant will flip the coordinates.
    Homographies that have close to 0 will also have undesirable behaviour (thin lines, extreme warping etc.).
    This function tries to identify these bad homographies.
    """
    if H.ndim > 2:
        det = np.linalg.det(H[:, :2, :2])
    else:
        det = np.linalg.det(H)
    return det > eps


def get_mosaic_dimensions(H: np.ndarray, width: Union[int, Sequence[int]], height: Union[int, Sequence[int]]) -> Sequence[int]:
    """
    Given a transformation homography, and the width and height of the input image. Calculate the bounding box of the warped image.
    """
    # Get the image corners
    dst_crn = get_corners(H, width, height)
    min_x, min_y = dst_crn.reshape(-1, 2).min(axis=0)
    max_x, max_y = dst_crn.reshape(-1, 2).max(axis=0)
    return min_x, min_y, max_x - min_x, max_y - min_y
    # Compute the top left and bottom right corners of bounding box
    return cv2.boundingRect(dst_crn.reshape(-1, 2).astype(np.float32))


def get_corners(H: np.ndarray, width: Union[int, Sequence[int]], height: Union[int, Sequence[int]]) -> np.ndarray:
    # Get the image corners
    if isinstance(width, int):
        src_crn = np.array([[[0, 0]],
                            [[width - 1, 0]],
                            [[width - 1, height - 1, ]],
                            [[0, height - 1]]], np.float32) + 0.5
    elif len(width):
        crns = [[[0, 0]]]
        for w, h in zip(width, height):
            crns.extend([[[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
        src_crn = np.array(crns, np.float32) + 0.5
    else:
        raise ValueError("width and height must either be ints or sequences of ints with same length.")
    if H.ndim > 2:
        src_crn = np.stack([src_crn] * len(H), 0)
    elif H.ndim == 2:
        src_crn = src_crn[None, ...]
        H = H[None, ...]
    N = src_crn.shape[1]
    src_crn_h = np.concatenate((src_crn, np.ones((len(H), N, 1, 1))), axis=-1)
    dst_crn_h = np.swapaxes(H @ np.swapaxes(src_crn_h.squeeze(axis=2), 1, 2), 1, 2)
    dst_crn = dst_crn_h[:, :, :2] / dst_crn_h[:, :, -1:]
    return dst_crn[:, :, None, :]


def alpha_blend_cuda(img1: cv2.cuda.GpuMat, img2: cv2.cuda.GpuMat, alpha_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """Perform alpha blending between two CUDA GpuMat images."""
    one_gpu = cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type(), 1)
    alpha_inv_gpu = cv2.cuda.subtract(one_gpu, alpha_gpu)

    if img1.type() != alpha_gpu.type():
        img1 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img1)) + (alpha_gpu,), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    if img2.type() != alpha_gpu.type():
        img2 = cv2.cuda.merge(tuple(cv2.cuda.normalize(channel, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F, cv2.cuda.GpuMat()) for channel in cv2.cuda.split(img2)) + (alpha_inv_gpu, ), cv2.cuda.GpuMat(alpha_gpu.size(), alpha_gpu.type()))

    blended = cv2.cuda.alphaComp(img1, img2, cv2.cuda.ALPHA_OVER)

    return cv2.cuda.cvtColor(blended, cv2.COLOR_BGRA2BGR)

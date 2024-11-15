import importlib.resources
import pickle
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Sequence

import cv2
import numpy as np
from numpy import typing as npt

import logging


logger = logging.getLogger(__name__)


@dataclass
class Node:
    """
    Data class to represent the record of a node of the ImageGraph.
    **Properties**
    - **identifier**: identifier for sql db lookup.
    - **name**: Name of the image.
    - **Dimensions**: Dimensions of the image.
    - **intrinsic_matrix**: Intrinsic matrix of the image.
    - **distortion**: Distortion model parameters of the image.
    """
    identifier: int
    name: str
    dimensions: tuple[int, int]
    intrinsic_matrix: Optional[npt.NDArray[float]] = None
    distortion: Optional[npt.NDArray[float]] = None

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.identifier == other.identifier
        return False


@dataclass
class DMatch:
    """
    Helper class to enable serializable representation of a cv2.DMatch object for pickling into sql dbs
    """
    queryIdx: int   # Index of the descriptor in the query set
    trainIdx: int   # Index of the descriptor in the train set
    imgIdx: int     # Index of the train image (in case of multiple images)
    distance: float # Distance between descriptors

    @classmethod
    def from_cv_dmatch(cls, dmatch: cv2.DMatch) -> 'DMatch':
        return cls(dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance)

    def to_cv_dmatch(self) -> cv2.DMatch:
        return cv2.DMatch(self.queryIdx, self.trainIdx, self.imgIdx, self.distance)


class DB(ABC):

    @abstractmethod
    def register_node(self, **kwargs) -> Node:
        ...

    @abstractmethod
    def node_exists(self, node: Node) -> bool:
        ...

    @abstractmethod
    def add_node_features(self, node: Node, features: dict[str, dict[str, Union[npt.NDArray[Union[np.uint8, np.float32]], Sequence[cv2.KeyPoint]]]]):
        ...

    @abstractmethod
    def get_node_features(self, node: Node) -> dict[str, dict[str, Union[npt.NDArray[Union[np.uint8, np.float32]], Sequence[cv2.KeyPoint]]]]:
        ...

    @abstractmethod
    def remove_node_features(self, node: Node) -> None:
        ...

    @abstractmethod
    def add_node_global_features(self, node: Node, global_features: npt.NDArray[float]):
        ...

    @abstractmethod
    def get_node_global_features(self, node: Node) -> npt.NDArray[float]:
        ...

    @abstractmethod
    def remove_node_global_features(self, node: Node) -> None:
        ...

    @abstractmethod
    def register_edge(self, node_from: Node, node_to: Node, overwrite: bool = False):
        ...

    @abstractmethod
    def edge_exists(self, node_from: Node, node_to: Node) -> bool:
        ...

    @abstractmethod
    def get_edge_matches(self, node_from: Node, node_to: Node) -> dict[str, Sequence[cv2.DMatch]]:
        ...

    @abstractmethod
    def add_edge_registration(self, node_from: Node, node_to: Node, registration: npt.NDArray[float]) -> None:
        ...

    @abstractmethod
    def get_edge_registration(self, node_from: Node, node_to: Node) -> npt.NDArray[float]:
        ...

    @abstractmethod
    def remove_node(self, node: Node) -> None:
        ...

    @abstractmethod
    def remove_edge(self, node_from: Node, node_to: Node) -> None:
        ...


class SQLDB(DB):
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._initialize_database()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # Ensure foreign keys are enforced
        return conn

    def _initialize_database(self):
        # Connect to the SQLite database (or create it if it doesn't exist)
        with self._connect() as conn:
            # Load the SQL schema template
            schema_sql = (importlib.resources.files('mosaicking.resources') / 'schema.sql').read_text()
            # Execute the SQL commands to create the database schema
            conn.executescript(schema_sql)
        logger.info(f"Database initialized at {self._db_path}")

    def register_node(self, name: str,
                      dimensions: tuple[int, int],
                      K: Optional[npt.NDArray[float]] = None,
                      D: Optional[npt.NDArray[float]] = None,
                      overwrite: bool = False) -> Node:
        """
        Here we have an image, we should already have a name, dimensions and maybe camera properties.
        Function to write out entry to schema db, and return the ImageNode object referencing that entry.
        :param name:
        :param dimensions:
        :param K:
        :param D:
        :param overwrite:
        :return:
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            # Convert arrays to blobs for SQL storage
            intrinsic_blob = pickle.dumps(K) if K is not None else None
            distortion_blob = pickle.dumps(D) if D is not None else None

            # Check if we need to overwrite an existing node
            if overwrite:
                cursor.execute("DELETE FROM Nodes WHERE name = ?", (name,))

            cursor.execute(
                """INSERT INTO Nodes (name, width, height, intrinsic_matrix, distortion)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO NOTHING""",
                (name, dimensions[0], dimensions[1], intrinsic_blob, distortion_blob)
            )
            conn.commit()

            # Retrieve and return the created node
            cursor.execute("SELECT id FROM Nodes WHERE name = ?", (name,))
            node_id = cursor.fetchone()[0]
            return Node(node_id, name, dimensions, K, D)

    def node_exists(self, node: Node) -> bool:
        """
        Check if a node exists in the database by its identifier.
        :param node: The node to check.
        :return: True if the node exists, False otherwise.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM Nodes WHERE id = ?", (node.identifier,))
            result = cursor.fetchone()
            return result is not None

    def add_node_features(self, node: Node, features: dict[str, dict[str, Union[npt.NDArray[Union[np.uint8, np.float32]], Sequence[cv2.KeyPoint]]]]):
        """
        Here we should have a node already registered in the db. Otherwise raise some kind of lookup error.
        We now want to write out to the features table.
        feature dict is structured as:
        {feature_type: {'keypoints': Union[cv2.KeyPoint | npt.NDArray[np.float32]],
                        'descriptors': npt.NDArray[Union[np.uint8 | np.float32]]}
        where feature_type is a specifying the type of feature extraction (e.g. 'ORB', 'SIFT' etc.)
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            for feature_type, feature_data in features.items():
                kps = feature_data.get('keypoints', None)
                if kps and isinstance(kps[0], cv2.KeyPoint):
                    kps = cv2.KeyPoint.convert(kps)
                keypoints_blob = pickle.dumps(kps)
                descriptors_blob = pickle.dumps(feature_data.get('descriptors', None))

                cursor.execute(
                    """INSERT INTO NodeFeatures (node_id, feature_type, keypoints, descriptors)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(node_id, feature_type) DO UPDATE
                       SET keypoints = excluded.keypoints, descriptors = excluded.descriptors""",
                    (node.identifier, feature_type, keypoints_blob, descriptors_blob)
                )
            conn.commit()

    def get_node_features(self, node: Node) -> dict[
        str, dict[str, Union[npt.NDArray[Union[np.uint8, np.float32]], Sequence[cv2.KeyPoint]]]]:
        """
        Retrieve the feature dictionary from the db. Raise ValueError if no features are found.
        :param node:
        :return:
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT feature_type, keypoints, descriptors FROM NodeFeatures WHERE node_id = ?",
                           (node.identifier,))
            rows = cursor.fetchall()

            if not rows:  # No features found, likely due to deletion or nonexistent node
                raise ValueError(f"No features found for node with ID {node.identifier}")

            # Process and return features if they exist
            features = {}
            for feature_type, kp_blob, desc_blob in rows:
                features[feature_type] = {
                    'keypoints': cv2.KeyPoint.convert(pickle.loads(kp_blob)),
                    'descriptors': pickle.loads(desc_blob)
                }
            return features

    def remove_node_features(self, node: Node) -> None:
        """
        Remove all features associated with a specific node from the database.
        :param node: The node whose features should be removed.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM NodeFeatures WHERE node_id = ?", (node.identifier,))
            conn.commit()

    def add_node_global_features(self, node: Node, global_features: dict[str, npt.NDArray[np.float32]]):
        with self._connect() as conn:
            cursor = conn.cursor()
            for feature_type, feature_data in global_features.items():
                descriptors_blob = pickle.dumps(feature_data)
                cursor.execute(
                    """INSERT INTO NodeGlobalFeatures (node_id, feature_type, global_features)
                       VALUES (?, ?, ?)
                       ON CONFLICT(node_id, feature_type) DO UPDATE
                       SET global_features = excluded.global_features""",
                    (node.identifier, feature_type, descriptors_blob)
                )
            conn.commit()

    def get_node_global_features(self, node: Node) -> dict[str, npt.NDArray[np.float32]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT feature_type, global_features FROM NodeGlobalFeatures WHERE node_id = ?",
                           (node.identifier,))
            rows = cursor.fetchall()

            if not rows:
                raise ValueError(f"No global features found for node with ID {node.identifier}")

            # Unpickle and return features
            features = {}
            for feature_type, desc_blob in rows:
                features.update({feature_type: pickle.loads(desc_blob)})
            return features

    def remove_node_global_features(self, node: Node) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM NodeGlobalFeatures WHERE node_id = ?", (node.identifier,))
            conn.commit()

    def register_edge(self, node_from: Node, node_to: Node, overwrite: bool = False):
        """
        Register a link between two existing nodes in the database.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            if overwrite:
                cursor.execute(
                    "DELETE FROM Edges WHERE source_id = ? AND target_id = ?",
                    (node_from.identifier, node_to.identifier)
                )

            cursor.execute(
                """INSERT INTO Edges (source_id, target_id)
                   VALUES (?, ?)
                   ON CONFLICT(source_id, target_id) DO NOTHING""",
                (node_from.identifier, node_to.identifier)
            )
            conn.commit()

    def edge_exists(self, node_from: Node, node_to: Node) -> bool:
        """
        Check if an edge exists in the database between two nodes.
        :param node_from: The source node of the edge.
        :param node_to: The target node of the edge.
        :return: True if the edge exists, False otherwise.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM Edges WHERE source_id = ? AND target_id = ?",
                           (node_from.identifier, node_to.identifier))
            result = cursor.fetchone()
            return result is not None

    def add_edge_matches(self, node_from: Node, node_to: Node, matches: dict[str, Sequence[cv2.DMatch]]):
        """
        Add match data between two existing nodes, organized by feature type.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            for feature_type, match_sequence in matches.items():
                # Convert each match to a serializable format
                serializable_matches = tuple(DMatch.from_cv_dmatch(match) for match in match_sequence)
                matches_blob = pickle.dumps(serializable_matches)

                cursor.execute(
                    """INSERT INTO EdgeMatches (source_id, target_id, feature_type, match_data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(source_id, target_id, feature_type) DO UPDATE
                       SET match_data = excluded.match_data""",
                    (node_from.identifier, node_to.identifier, feature_type, matches_blob)
                )
            conn.commit()

    def get_edge_matches(self, node_from: Node, node_to: Node) -> dict[str, Sequence[cv2.DMatch]]:
        """
        Retrieve match data between two existing nodes, organized by feature type.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT feature_type, match_data FROM EdgeMatches WHERE source_id = ? AND target_id = ?",
                (node_from.identifier, node_to.identifier)
            )
            results = cursor.fetchall()

            if not results:
                raise ValueError("Matches not found for the specified edge.")

            # Deserialize and organize matches by feature type
            matches_by_type = {}
            for feature_type, match_data_blob in results:
                matches_by_type[feature_type] = [
                    dmatch.to_cv_dmatch() for dmatch in pickle.loads(match_data_blob)
                ]
            return matches_by_type

    def add_edge_registration(self, node_from: Node, node_to: Node, registration: npt.NDArray[float],
                              reproj_error: float | None = None, frac_inliers: float | None = None) -> None:
        """
        Add or update a calculated registration to an existing edge.
        :param node_from: Source node.
        :param node_to: Target node.
        :param registration: Registration matrix.
        :param reproj_error: euclidean reprojection error (pix)
        :param frac_inliers: Fraction of inlier points used in estimation
        """
        registration_blob = pickle.dumps(registration)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO EdgeRegistration (source_id, target_id, registration, reproj_error, frac_inliers)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_id, target_id) DO UPDATE
                   SET registration = excluded.registration""",
                (node_from.identifier, node_to.identifier, registration_blob, reproj_error, frac_inliers)
            )
            conn.commit()

    def get_edge_registration(self, node_from: Node, node_to: Node) -> tuple[npt.NDArray[float], float | None, float | None]:
        """
        Fetch the registration for an existing edge.
        :param node_from: Source node.
        :param node_to: Target node.
        :return: Registration matrix as a NumPy array.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT registration, reproj_error, frac_inliers FROM EdgeRegistration WHERE source_id = ? AND target_id = ?",
                (node_from.identifier, node_to.identifier)
            )
            result = cursor.fetchone()

            if result is None:
                raise ValueError("Registration data not found for the specified edge.")

            # Deserialize and return the registration matrix
            return pickle.loads(result[0]), result[1], result[2]

    def remove_node(self, node: Node) -> None:
        """
        Remove a node and associated data from the database.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Delete the node itself
            cursor.execute("DELETE FROM Nodes WHERE id = ?", (node.identifier,))
            conn.commit()

    def remove_edge(self, node_from: Node, node_to: Node) -> None:
        """
        Remove an edge and associated data from the database.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Delete the edge itself from Edges table
            cursor.execute("DELETE FROM Edges WHERE source_id = ? AND target_id = ?",
                           (node_from.identifier, node_to.identifier))
            conn.commit()

import random
import unittest

from mosaicking.core import db
from mosaicking.core import interface
import cv2
import numpy as np
from numpy import testing as nptest
from pathlib import Path
import tempfile


class TestCoreDB(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file for each test
        self.db_path = Path(tempfile.NamedTemporaryFile(delete=False).name)
        self.db = db.SQLDB(self.db_path)

    def tearDown(self):
        # Clean up the temporary database file
        self.db_path.unlink()

    def test_add_node(self):
        # Add a node to the database
        name = "image_1"
        dimensions = (800, 600)
        node = self.db.register_node(name=name, dimensions=dimensions)

        # Verify the node has been added correctly
        self.assertEqual(node.name, name)
        self.assertEqual(node.dimensions, dimensions)

        # Features should be non-existent (ValueError)
        with self.assertRaises(ValueError):
            self.db.get_node_features(node)


    def test_add_node_features(self):
        # Add a node to the database
        name = "image_X"
        dimensions = (800, 600)
        node = self.db.register_node(name=name, dimensions=dimensions)
        features = {"feature_1": {'descriptors': np.random.randint(0, 255, size=(1000, 32)),
                                  'keypoints': tuple(cv2.KeyPoint(random.random(),
                                                                  random.random(),
                                                                  random.random(),
                                                                  random.random(),
                                                                  random.random(),
                                                                  random.randint(0, 15),
                                                                  random.randint(0, 15)) for _ in range(1000))
                                  }
                    }
        self.db.add_node_features(node, features)
        retrieved_features = self.db.get_node_features(node)
        for feature_type in features:
            # all feature types must exist
            self.assertIn(feature_type, retrieved_features)
            # all descriptors must match
            nptest.assert_array_equal(features[feature_type]['descriptors'], retrieved_features[feature_type]['descriptors'])
            # all keypoints must match
            nptest.assert_array_equal(cv2.KeyPoint.convert(features[feature_type]['keypoints']),
                                      cv2.KeyPoint.convert(retrieved_features[feature_type]['keypoints']))

    def test_add_edge(self):
        # Add two nodes to link them with an edge
        node1 = self.db.register_node(name="image_1", dimensions=(800, 600))
        node2 = self.db.register_node(name="image_2", dimensions=(800, 600))

        # Register an edge between node1 and node2
        self.db.register_edge(node_from=node1, node_to=node2)

        self.assertTrue(self.db.edge_exists(node1, node2))

        # Attempt to retrieve matches for the edge (should be empty initially)
        with self.assertRaises(ValueError):
            self.db.get_edge_matches(node1, node2)  # Expecting a ValueError as no matches are added

    def test_add_edge_matches(self):
        # Add nodes and create an edge
        node1 = self.db.register_node(name="image_1", dimensions=(800, 600))
        node2 = self.db.register_node(name="image_2", dimensions=(800, 600))
        self.db.register_edge(node_from=node1, node_to=node2)

        # Create some dummy matches
        matches = [db.DMatch(queryIdx=0, trainIdx=1, imgIdx=2, distance=0.5),
                   db.DMatch(queryIdx=1, trainIdx=2, imgIdx=2, distance=1.2)]

        # Add matches to the edge
        self.db.add_edge_matches(node1, node2, dict(test=matches))

        # Retrieve matches and verify they match the added data
        retrieved_matches = self.db.get_edge_matches(node1, node2)
        self.assertEqual(len(retrieved_matches["test"]), len(matches))
        for orig, ret in zip(matches, retrieved_matches["test"]):
            self.assertEqual(orig.queryIdx, ret.queryIdx)
            self.assertEqual(orig.trainIdx, ret.trainIdx)
            self.assertAlmostEqual(orig.distance, ret.distance, places=5)

    def test_delete_node_and_cascade(self):
        # Register nodes and an edge between them
        node1 = self.db.register_node(name="image_1", dimensions=(800, 600))
        node2 = self.db.register_node(name="image_2", dimensions=(1024, 768))
        self.db.register_edge(node_from=node1, node_to=node2)

        # Add features to node1 and matches to the edge
        features = {"feature_1": {'descriptors': np.random.randint(0, 255, (500, 32)),
                                  'keypoints': tuple(
                                      cv2.KeyPoint(random.random(), random.random(), random.random()) for _ in
                                      range(500))
                                  }}
        self.db.add_node_features(node1, features)
        matches = [db.DMatch(queryIdx=0, trainIdx=1, imgIdx=2, distance=0.5)]
        self.db.add_edge_matches(node_from=node1, node_to=node2, matches={"feature_1": matches})

        # Check data presence before deletion
        node1_features = self.db.get_node_features(node1)
        self.assertTrue(node1_features)  # Confirm features were added for node1
        edge_matches = self.db.get_edge_matches(node_from=node1, node_to=node2)
        self.assertTrue(edge_matches)  # Confirm matches were added for the edge

        # Delete node1 and check cascading deletes
        self.db.remove_node(node1)

        # Verify node1 is deleted and related data cascaded
        self.assertFalse(self.db.node_exists(node1))
        with self.assertRaises(ValueError):
            self.db.get_edge_matches(node_from=node1, node_to=node2)
        with self.assertRaises(ValueError):
            self.db.get_node_features(node1)

        # Verify node2 still exists and doesn't have features
        self.assertTrue(self.db.node_exists(node2))
        with self.assertRaises(ValueError):
            self.db.get_node_features(node2)

    def test_delete_edge_only(self):
        # Register nodes and an edge between them
        node1 = self.db.register_node(name="image_A", dimensions=(800, 600))
        node2 = self.db.register_node(name="image_B", dimensions=(1024, 768))
        self.db.register_edge(node_from=node1, node_to=node2)

        # Add matches to the edge
        matches = [db.DMatch(queryIdx=0, trainIdx=1, imgIdx=2, distance=0.5)]
        self.db.add_edge_matches(node_from=node1, node_to=node2, matches={"feature_1": matches})

        # Delete the edge and verify cascading delete of edge attributes
        self.db.remove_edge(node_from=node1, node_to=node2)
        with self.assertRaises(ValueError):
            self.db.get_edge_matches(node_from=node1, node_to=node2)

        # Verify that nodes are not deleted
        self.assertTrue(self.db.node_exists(node1))
        self.assertTrue(self.db.node_exists(node2))

    def test_delete_node_features(self):
        # Register a node and add features
        node = self.db.register_node(name="image_X", dimensions=(800, 600))
        features = {"feature_1": {'descriptors': np.random.randint(0, 255, (500, 32)),
                                  'keypoints': tuple(
                                      cv2.KeyPoint(random.random(), random.random(), random.random()) for _ in
                                      range(500))}
                    }
        self.db.add_node_features(node, features)

        # Remove node features manually
        self.db.remove_node_features(node)

        # Verify that features are deleted but the node still exists
        self.assertTrue(self.db.node_exists(node))
        with self.assertRaises(ValueError):
            self.db.get_node_features(node)

    def test_cascade_delete_of_edge_registration_and_matches(self):
        # Register nodes, an edge, and add matches and registration data
        node1 = self.db.register_node(name="image_1", dimensions=(800, 600))
        node2 = self.db.register_node(name="image_2", dimensions=(1024, 768))
        self.db.register_edge(node_from=node1, node_to=node2)

        matches = [db.DMatch(queryIdx=0, trainIdx=1, imgIdx=2, distance=0.5)]
        self.db.add_edge_matches(node_from=node1, node_to=node2, matches={"feature_1": matches})

        registration = np.random.rand(3, 3)
        self.db.add_edge_registration(node_from=node1, node_to=node2, registration=registration)

        # Delete node1 and ensure edge registration and matches are deleted
        self.db.remove_node(node1)
        with self.assertRaises(ValueError):
            self.db.get_edge_registration(node_from=node1, node_to=node2)
        with self.assertRaises(ValueError):
            self.db.get_edge_matches(node_from=node1, node_to=node2)


class TestImageGraph(unittest.TestCase):
    def setUp(self):
        # Create an ImageGraph instance with a temporary database directory
        self.graph = interface.ImageGraph(db_dir=Path(tempfile.mkdtemp()))

        # Define sample node properties
        self.node1 = self.graph.register_node(
            name="image_1", dimensions=(800, 600),
            K=np.eye(3), D=np.zeros(5), winky=";)"
        )
        self.node2 = self.graph.register_node(
            name="image_2", dimensions=(1024, 768),
            K=np.eye(3), D=np.zeros(5), winky=";)"
        )

    def test_register_node(self):
        # Check if the node is registered in the graph and backend
        self.assertIn(self.node1, self.graph.nodes())
        self.assertIn(self.node2, self.graph.nodes())

        # Verify metadata retrieval
        nodes_with_data = self.graph.nodes(data=True)
        self.assertGreater(len(nodes_with_data), 0)
        self.assertTrue(any(data for node, data in nodes_with_data))

    def test_add_features(self):
        # Add features to node1 and retrieve them
        features = {
            "ORB": {
                "keypoints": (cv2.KeyPoint(x=150.0, y=200.0, size=1.0),),
                "descriptors": np.random.randint(0, 255, size=(1, 32), dtype=np.uint8)
            }
        }
        self.graph.add_features(self.node1, features)

        # Retrieve and verify the features from the backend
        retrieved_features = self.graph._db_backend.get_node_features(self.node1)
        self.assertIn("ORB", retrieved_features)
        np.testing.assert_array_equal(
            features["ORB"]["descriptors"],
            retrieved_features["ORB"]["descriptors"]
        )

        # Retrieve and very the features from nodes
        # Verify metadata retrieval
        nodes_with_data = self.graph.nodes(data=True)
        self.assertGreater(len(nodes_with_data), 0)

    def test_register_edge(self):
        # Register an edge and verify it exists in the graph and backend
        self.graph.register_edge(self.node1, self.node2)
        self.assertIn((self.node1, self.node2), self.graph.edges())

    def test_add_matches(self):
        # Register an edge and add matches between node1 and node2
        self.graph.register_edge(self.node1, self.node2)

        matches = {"whump": [
            cv2.DMatch(0, 1, 2, 0.5),
            cv2.DMatch(1, 2, 2, 1.2)
        ]}
        self.graph.add_matches(self.node1, self.node2, matches)

        # Retrieve matches from the backend and verify them
        retrieved_matches = self.graph._db_backend.get_edge_matches(self.node1, self.node2)
        self.assertEqual(len(retrieved_matches['whump']), len(matches['whump']))
        for orig, ret in zip(matches['whump'], retrieved_matches['whump']):
            self.assertEqual(orig.queryIdx, ret.queryIdx)
            self.assertEqual(orig.trainIdx, ret.trainIdx)
            self.assertAlmostEqual(orig.distance, ret.distance, places=5)

    def tearDown(self):
        # Remove the temporary directory and database
        for child in self.graph.db_path.parent.iterdir():
            child.unlink()
        self.graph.db_path.parent.rmdir()

# TODO: These tests are redundant. Need to adjust for dealing with pickling / de-pickling of data.
# class TestCore(unittest.TestCase):
#
#     def test_pickling_node(self):
#         features = {"test_feature": {"keypoints": [cv2.KeyPoint(x=0.1, y=0.1, size=10.0),
#                                                    cv2.KeyPoint(x=0.2, y=0.2, size=20.0)],
#                                      "descriptors": np.random.randint(255, size=(2, 128), dtype=np.uint8), }}
#         cls = db.Node(features=features, dimensions=(300, 200))
#         self.assertTrue(isinstance(cls.features["test_feature"]["keypoints"], np.ndarray))
#
#     def test_pickling_edge(self):
#         graph = interface.ImageGraph(Path('./data'))
#         features = {"test_feature": {"keypoints": [cv2.KeyPoint(x=0.1, y=0.1, size=10.0),
#                                                    cv2.KeyPoint(x=0.2, y=0.2, size=20.0)],
#                                      "descriptors": np.random.randint(255, size=(2, 128), dtype=np.uint8), }}
#         n1 = db.Node(features=features, dimensions=(300, 200))
#         n2 = db.Node(features=features, dimensions=(200, 300))
#         matches = {"test_feature": (cv2.DMatch(1, 2, 3, 42),
#                                     cv2.DMatch(7, 8, 9, 10))}
#         edge = db.Edge(matches)
#         graph.add_image(0, n1)
#         graph.add_image(1, n2)
#         graph.add_registration(0, 1, edge)
#         edge2 = db.Edge()
#
#     def test_imagenode_dataclass_asdict(self):
#         features = {"test_feature": {"keypoints": [cv2.KeyPoint(x=0.1, y=0.1, size=10.0),
#                                   cv2.KeyPoint(x=0.2, y=0.2, size=20.0)],
#                     "descriptors": np.random.randint(255, size=(2, 128), dtype=np.uint8),}}
#         graph = interface.ImageGraph(Path('./data'))
#         cls = db.Node(features=features)
#         self.assertTrue(isinstance(cls.features["test_feature"]["keypoints"], np.ndarray))

if __name__ == '__main__':
    unittest.main()

import sqlite3
import tempfile
import unittest
from pathlib import Path

class TestCascadingDeletes(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file for each test
        self.db_path = Path(tempfile.NamedTemporaryFile(delete=False).name)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support

        # Define tables as per schema
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS Nodes (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS NodeFeatures (
                node_id INTEGER NOT NULL,
                feature_type TEXT NOT NULL,
                keypoints BLOB,
                descriptors BLOB,
                FOREIGN KEY(node_id) REFERENCES Nodes(id) ON DELETE CASCADE,
                UNIQUE(node_id, feature_type)
            );

            CREATE TABLE IF NOT EXISTS Edges (
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                FOREIGN KEY(source_id) REFERENCES Nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES Nodes(id) ON DELETE CASCADE,
                PRIMARY KEY (source_id, target_id)
            );
        """)

    def tearDown(self):
        # Close and remove the temporary database file
        self.conn.close()
        self.db_path.unlink()

    def test_node_deletion_cascade(self):
        # Insert nodes and related data to test cascading delete
        cursor = self.conn.cursor()

        # Insert two nodes
        cursor.execute("INSERT INTO Nodes (name, width, height) VALUES ('node1', 800, 600)")
        node1_id = cursor.lastrowid
        cursor.execute("INSERT INTO Nodes (name, width, height) VALUES ('node2', 1024, 768)")
        node2_id = cursor.lastrowid

        # Insert features for node1
        cursor.execute("INSERT INTO NodeFeatures (node_id, feature_type) VALUES (?, 'feature_1')", (node1_id,))

        # Insert an edge between node1 and node2
        cursor.execute("INSERT INTO Edges (source_id, target_id) VALUES (?, ?)", (node1_id, node2_id))

        # Commit the inserts
        self.conn.commit()

        # Delete node1 and check if cascading delete works
        cursor.execute("DELETE FROM Nodes WHERE id = ?", (node1_id,))
        self.conn.commit()

        # Check that related entries in NodeFeatures and Edges were deleted
        cursor.execute("SELECT * FROM NodeFeatures WHERE node_id = ?", (node1_id,))
        features = cursor.fetchall()
        self.assertEqual(features, [])  # Should be empty if cascading delete worked

        cursor.execute("SELECT * FROM Edges WHERE source_id = ? OR target_id = ?", (node1_id, node1_id))
        edges = cursor.fetchall()
        self.assertEqual(edges, [])  # Should be empty if cascading delete worked

if __name__ == '__main__':
    unittest.main()

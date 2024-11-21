
-- Nodes table, representing individual nodes
CREATE TABLE IF NOT EXISTS Nodes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    intrinsic_matrix BLOB,
    distortion BLOB
);

-- NodeFeatures table for storing attributes of individual nodes
CREATE TABLE IF NOT EXISTS NodeFeatures (
    node_id INTEGER NOT NULL,
    feature_type TEXT NOT NULL,
    keypoints BLOB,
    descriptors BLOB,
    FOREIGN KEY(node_id) REFERENCES Nodes(id) ON DELETE CASCADE,
    UNIQUE(node_id, feature_type)
);

-- NodeGlobalFeatures table for storing attributes of individual nodes
CREATE TABLE IF NOT EXISTS NodeGlobalFeatures (
    node_id INTEGER NOT NULL,
    feature_type TEXT NOT NULL,
    global_features BLOB,
    FOREIGN KEY(node_id) REFERENCES Nodes(id) ON DELETE CASCADE,
    UNIQUE(node_id, feature_type)
);

-- Edges table with separate cascading deletes for source_id and target_id
CREATE TABLE IF NOT EXISTS Edges (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    FOREIGN KEY(source_id) REFERENCES Nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES Nodes(id) ON DELETE CASCADE,
    PRIMARY KEY (source_id, target_id)
);

-- EdgeMatches table for storing edge attribute data by feature type
CREATE TABLE IF NOT EXISTS EdgeMatches (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    feature_type TEXT NOT NULL,
    match_data BLOB,
    FOREIGN KEY(source_id, target_id) REFERENCES Edges(source_id, target_id) ON DELETE CASCADE,
    UNIQUE(source_id, target_id, feature_type)
);

-- EdgeRegistration table for storing registration data for each edge
CREATE TABLE IF NOT EXISTS EdgeRegistration (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    registration BLOB,
    reproj_error FLOAT,
    frac_inliers FLOAT,
    FOREIGN KEY(source_id, target_id) REFERENCES Edges(source_id, target_id) ON DELETE CASCADE,
    UNIQUE(source_id, target_id)
);

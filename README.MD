# mosaic-library #

This is a python library for manipulating images for underwater mosaicking applications. Currently, this libary is 
capable of:

1. Reading input videos.
2. Basic pre-processing of images, including undistorting, crop & scaling, color, contrast, and lighting balancing as well as detail enhancement.
3. Image registration, such as feature detection and description, matching, and transformation estimation in similarity, affine, and perspective cases.
4. Infinite and finite homography transformations on imagery given extrinsic rotations and translations between the camera and a new "virtual" camera.
5. Camera rotation compensation from orientation data (e.g. IMU).
6. Sequential mosaicking map generation, with minimal memory management support (database + tiling).

The [mosaic-library](https://pypi.org/project/mosaic-library/) covers these two use-cases: 

1. Basic implementation (you need an opencv-contrib backend): `pip install mosaic-library[opencv]`
2. Custom implementation (you have your own opencv-contrib build): `pip install mosaic-library`

See the [examples](https://github.com/DTUAqua-ObsTek/mosaic-library/tree/main/examples) folder for demonstrations of the mosaic module.

## Version 2.0.0 release candidate

Checkout out v2.0.0-rc2 for the most recent pre-release tag.

### Release Notes
1. Classes for the __mosaicking.mosaic__ module. Now you can implement your own classes for mosaicking operations. See __mosaicking.mosaic.Mosaic__ and __mosaicking.mosaic.SequentialMosaic__ for templates.
2. Classes for feature extraction: Use the __mosaicking.registration.FeatureDetector__ class for implementing your own feature detectors.
3. Classes for preprocessing operations: Use the __mosaicking.preprocessing.Preprocessor__ class for implementing your own preprocessing methods.
4. Classes for data reading operations: Use the __mosaicking.utils.DataReader__ class to implement your own data schema.
4. Support for NVIDIA CUDA and Video Codec SDK. __mosaic-library__ now takes advantage of OpenCV builds with CUDA and NVCUVID flags.
5. See the docker build file and compose files for building a working dev container with cuda or cudacodec support on your system (linux only).
6. Implementation of SQLlite: large video files cause memory issues for mosaicking v1. We now cache the feature extraction and matching components to a database file to decrease RAM consumption, allow multiple video sources to be used in the same mosaic, allow incremental updates.
7. Use of __networkx__: mosaics can be represented as transformational relationships between images, we use [networkx](https://networkx.org/) to represent those relationships through node (local and global image features) and edge attributes (matches and homography transformations).
8. Use of __scikit-learn__: extraction of global image features through Visual Bag of Words is supported through __sklearn__'s k-means clustering.
9. Better handling of poor feature matching / homography estimation (see __mosaicking.mosaic.SequentialMosaic__).
10. Basic sphinx documentation hosted on github pages.

## Roadmap ##

1. Implementation of homography graph optimization strategies (loop closure + GTSAM / g2o) 
2. Composite data sources: merge multiple image data sources into one mosaic.
3. Integration of additional navigation data sources (USBL, DVL).

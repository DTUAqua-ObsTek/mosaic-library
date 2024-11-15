from mosaicking import mosaic
import logging
from pathlib import Path


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_path = Path(f"../data/mosaicking/fishes.mp4").resolve()
    output_path = Path(f"../fishes_output/{data_path.stem}_output").resolve()
    mosaic = mosaic.SequentialMosaic(   data_path=data_path,
                                        project_path=output_path,
                                        feature_types=("ORB",),
                                        extractor_kwargs={"nFeatures": 4000},
                                        keypoint_roi=True,
                                        epsilon=1e-3,
                                        overwrite=False,
                                        alpha=1.0,
                                        )
    mosaic.extract_features()
    mosaic.match_features()
    mosaic.registration()
    mosaic.global_registration()
    mosaic.generate((4096, 4096))

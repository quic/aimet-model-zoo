import pytest
from pathlib import Path


@pytest.fixture(scope='module')
def data_folder(tmp_path_factory):
    """
    This fixture will return a shortcut path for saved data. When there's no
    such shortcut path, a tmp path will be generated. Use this with discretion
    because anything stored in a tmp path will be gone. Set DEPENDENCY_DATA_PATH
    only when you have permanent files stored for testing
    """
    if is_cache_env_set():
        dependency_path = Path(os.getenv('DEPENDENCY_DATA_PATH'))
    else:
        dependency_path = None

    data_path = dependency_path if (dependency_path and dependency_path.exists()) else tmp_path_factory.mktemp('data')

    yield data_path


@pytest.fixture(autouse=True)
def dataset_path(data_path):
    """this fixture return the dataset paths for acceptance tests"""

    dataset_path = {
               "image_classification":str(data_path)+"ILSVRC2012_PyTorch_reduced/val",
               "object_detection":str(data_path)+"tiny_coco/val_2017",
               "pose_estimation":str(data_path)+"tiny_coco/val_2017",
               "semantic_segmentation":str(data_path)+"cityscapes",
               "super_reslution":str(data_path)+"/super_resolution_data/Set5/image_SRF_4_HR"
               }
    return dataset_path 

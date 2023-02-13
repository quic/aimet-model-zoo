#pylint: skip-file
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
#  Changes from QuIC are licensed under the terms and conditions at 
#  https://github.com/quic/aimet-model-zoo/blob/develop/LICENSE.pdf"
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

import pyarrow as pa

import datasets
from datasets.tasks import ImageClassification


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class ImageFolderConfig(datasets.BuilderConfig):
    """BuilderConfig for ImageFolder."""

    features: Optional[datasets.Features] = None

    @property
    def schema(self):
        return pa.schema(self.features.type) if self.features is not None else None

class ImageFolder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ImageFolderConfig

    def _info(self):
        root_dir = self.config.data_dir
        self.config.data_files = self.config.data_dir

        self.classes, self.class_to_idx = self._find_classes_from_dir()

        return datasets.DatasetInfo(
            features=datasets.Features(
                {"image_file_path": datasets.Value("string"), "labels": datasets.features.ClassLabel(names=self.classes)}
            ),
            task_templates=[
                datasets.tasks.ImageClassification(
                    image_column="image_file_path", label_column="labels")
            ]
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")

        data_files = self.config.data_files
        if isinstance(data_files, str):
            folder = data_files
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"archive_path": folder})]
        splits = []
        for split_name, folder in data_files.items():
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"archive_path": folder}))
        return splits

    def _generate_examples(self, archive_path):
        data_count = 0
        for target_class in self.class_to_idx.keys():
            class_index = str(self.class_to_idx[target_class])
            target_dir = os.path.join(archive_path, target_class)
            for _root, _, _fnames in os.walk(target_dir):
                for fname in sorted(_fnames):
                    if fname.lower().endswith('jpg') or fname.lower().endswith('jpeg'):
                        yield data_count, {"image_file_path": os.path.join(_root, fname), 
                                "labels": target_class}
                        data_count += 1
    
    def _find_classes_from_dir(self, directory=None):
        if directory is None:
            if isinstance(self.config.data_dir, str):
                directory = self.config.data_dir
            else:
                # assume data_dir is dick
                if "train" in self.config.data_dir:
                    directory = self.config.data_dir["train"]
                else:
                    for key in self.config.data_dir.keys():
                        directory = self.config.data_dir[key]
                        break

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

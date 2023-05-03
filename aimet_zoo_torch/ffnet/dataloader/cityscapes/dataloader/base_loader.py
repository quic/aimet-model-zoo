# pylint: skip-file
"""
Copyright 2020 Nvidia Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
import os.path as path
import glob
import numpy as np
import torch

from PIL import Image
from torch.utils import data
from aimet_zoo_torch.ffnet.model.config import CITYSCAPES_IGNORE_LABEL, CITYSCAPES_NUM_CLASSES#, cityscapes_base_path
from ..utils.misc import tensor_to_pil
from ..cityscapes import find_directories
from .. import cityscapes_labels
from scipy.ndimage.morphology import distance_transform_edt


class BaseLoader(data.Dataset):
    def __init__(self, quality, mode, joint_transform_list, img_transform, label_transform):

        super(BaseLoader, self).__init__()
        self.quality = quality
        self.mode = mode
        self.joint_transform_list = joint_transform_list
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.train = mode == "train"
        self.id_to_trainid = {}
        self.all_imgs = None

    @staticmethod
    def find_images(img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.
        """
        img_path = "{}/*.{}".format(img_root, img_ext)
        imgs = glob.glob(img_path)
        items = []
        for full_img_fn in imgs:
            img_dir, img_fn = os.path.split(full_img_fn)
            img_name, _ = os.path.splitext(img_fn)
            full_mask_fn = "{}.{}".format(img_name, mask_ext)
            full_mask_fn = os.path.join(mask_root, full_mask_fn)
            assert os.path.exists(full_mask_fn)
            items.append((full_img_fn, full_mask_fn))
        return items

    def colorize_mask(self, image_array):
        """
        Colorize the segmentation mask
        """
        new_mask = Image.fromarray(image_array.astype(np.uint8)).convert("P")
        new_mask.putpalette(self.color_mapping)
        return new_mask

    def dump_images(self, img_name, mask, class_id, img):
        img = tensor_to_pil(img)
        outdir = "new_dump_imgs_{}".format(self.mode)
        os.makedirs(outdir, exist_ok=True)
        dump_img_name = img_name
        out_img_fn = os.path.join(outdir, dump_img_name + ".png")
        out_msk_fn = os.path.join(outdir, dump_img_name + "_mask.png")
        out_raw_fn = os.path.join(outdir, dump_img_name + "_mask_raw.png")
        mask_img = self.colorize_mask(np.array(mask))
        raw_img = Image.fromarray(np.array(mask))
        img.save(out_img_fn)
        mask_img.save(out_msk_fn)
        raw_img.save(out_raw_fn)

    def do_transforms(self, img, mask, img_name):
        """
        Do transformations to image and mask
        :returns: image, mask
        """
        scale_float = 1.0

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                outputs = xform(img, mask)
                if len(outputs) == 3:
                    img, mask, scale_float = outputs
                else:
                    img, mask = outputs

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)

        return img, mask, scale_float

    def read_images(self, img_path, mask_path, mask_out=False):
        img = Image.open(img_path).convert("RGB")
        if mask_path is None or mask_path == "":
            w, h = img.size
            mask = np.zeros((h, w))
        else:
            mask = Image.open(mask_path)

        drop_out_mask = None
        # This code is specific to cityscapes

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)

        mask = mask.copy()
        for k, v in self.id_to_trainid.items():
            binary_mask = mask == k
            mask[binary_mask] = v

        mask = Image.fromarray(mask.astype(np.uint8))
        return img, mask, img_name

    def mask_to_onehot(self, mask, num_classes):
        _mask = [mask == (i + 1) for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)

    def onehot_to_binary_edges(self, mask, radius, num_classes):
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
        edgemap = np.zeros(mask.shape[1:])
        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(
                1.0 - mask_pad[i, :]
            )
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

    def __getitem__(self, index):
        """
        Generate data:
        :return:
        - image: image, tensor
        - mask: mask, tensor
        - image_name: basename of file, string
        """
        img_path, mask_path = self.all_imgs[index]

        mask_out = False

        img, mask, img_name = self.read_images(img_path, mask_path, mask_out=mask_out)

        ######################################################################
        # Thresholding is done when using coarse-labelled Cityscapes images
        ######################################################################

        img, mask, scale_float = self.do_transforms(img, mask, img_name)

        _edgemap = mask.numpy()
        _edgemap = self.mask_to_onehot(_edgemap, self.num_classes)
        _edgemap = self.onehot_to_binary_edges(_edgemap, 2, self.num_classes)
        edgemap = torch.from_numpy(_edgemap).float()

        return img, mask, edgemap, img_name, scale_float

    def __len__(self):
        return len(self.all_imgs)


class Cityscapes(BaseLoader):
    num_classes = CITYSCAPES_NUM_CLASSES
    ignore_label = CITYSCAPES_IGNORE_LABEL
    trainid_to_name = {}
    color_mapping = []

    def __init__(
        self,
        mode,
        quality="fine",
        joint_transform_list=None,
        img_transform=None,
        label_transform=None,
        eval_folder=None,
        cityscapes_base_path=None
    ):

        super(Cityscapes, self).__init__(
            quality=quality,
            mode=mode,
            joint_transform_list=joint_transform_list,
            img_transform=img_transform,
            label_transform=label_transform,
        )

        self.root = cityscapes_base_path
        self.id_to_trainid = cityscapes_labels.label2trainid
        self.trainid_to_name = cityscapes_labels.trainId2name

        self.fill_colormap()
        img_ext = "png"
        mask_ext = "png"
        img_root = path.join(self.root, "leftImg8bit")
        mask_root = path.join(self.root, "gtFine")

        self.fine_cities = find_directories(self.root)
        self.all_imgs = self.find_cityscapes_images(
            self.fine_cities, img_root, mask_root, img_ext, mask_ext
        )

    def find_cityscapes_images(self, cities, img_root, mask_root, img_ext, mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        cities: a list of cities, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []
        for city in cities:
            img_dir = "{root}/{city}".format(root=img_root, city=city)
            for file_name in os.listdir(img_dir):
                basename, ext = os.path.splitext(file_name)
                assert ext == "." + img_ext, "{} {}".format(ext, img_ext)
                full_img_fn = os.path.join(img_dir, file_name)
                basename, ext = file_name.split("_leftImg8bit")
                mask_fn = f"{basename}_gtFine_labelIds{ext}"
                full_mask_fn = os.path.join(mask_root, city, mask_fn)
                if os.path.isfile(full_mask_fn):
                    items.append((full_img_fn, full_mask_fn))

        print("Running Inference on {} samples".format(len(items)))

        return items

    def fill_colormap(self):
        palette = [
            128,
            64,
            128,
            244,
            35,
            232,
            70,
            70,
            70,
            102,
            102,
            156,
            190,
            153,
            153,
            153,
            153,
            153,
            250,
            170,
            30,
            220,
            220,
            0,
            107,
            142,
            35,
            152,
            251,
            152,
            70,
            130,
            180,
            220,
            20,
            60,
            255,
            0,
            0,
            0,
            0,
            142,
            0,
            0,
            70,
            0,
            60,
            100,
            0,
            80,
            100,
            0,
            0,
            230,
            119,
            11,
            32,
        ]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette

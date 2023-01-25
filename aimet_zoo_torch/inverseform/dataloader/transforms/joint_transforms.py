"""
# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
#
#
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
import math
import numbers
from PIL import Image, ImageOps
import numpy as np
import random
from aimet_zoo_torch.inverseform.model.utils.config import cfg


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def add_margin(pil_img, top, right, bottom, left, margin_color):
    """
    Add margin around an image

    top, right, bottom, left are the margin widths, in pixels
    margin_color is what to use for the margins
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), margin_color)
    result.paste(pil_img, (left, top))
    return result


def set_crop_size(crop_size):
    if isinstance(crop_size, (list, tuple)):
        size = crop_size
    elif isinstance(crop_size, numbers.Number):
        size = (int(crop_size), int(crop_size))
    else:
        raise
    return size


class RandomCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.


    if cfg.DATASET.TRANSLATION_AUG_FIX is set, we insure that there's always
    translation randomness of at least that value around the image.

    if image < crop_size:
        # slide crop within image, random offset
    else:
        # slide image within crop
    """
    def __init__(self, crop_size, nopad=True):
        self.size = set_crop_size(crop_size)
        self.ignore_index = cfg.DATASET.IGNORE_LABEL
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    @staticmethod
    def crop_in_image(centroid, target_w, target_h, w, h, img, mask):
        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - target_w
            max_y = h - target_h
            x1 = random.randint(c_x - target_w, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - target_h, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == target_w:
                x1 = 0
            else:
                x1 = random.randint(0, w - target_w)
            if h == target_h:
                y1 = 0
            else:
                y1 = random.randint(0, h - target_h)

        return [img.crop((x1, y1, x1 + target_w, y1 + target_h)),
                mask.crop((x1, y1, x1 + target_w, y1 + target_h))]

    def image_in_crop(self, target_w, target_h, w, h, img, mask):
        # image smaller than crop, so slide image within crop
        x_total_margin = target_w - w
        y_total_margin = target_h - h

        left = random.randint(0, x_total_margin)
        right = x_total_margin - left

        top = random.randint(0, y_total_margin)
        bottom = y_total_margin - top

        slid_image = add_margin(img, top, right, bottom, left,
                                self.pad_color)
        slid_mask = add_margin(mask, top, right, bottom, left,
                               self.ignore_index)
        return [slid_image, slid_mask]

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size
        w, h = img.size
        target_h, target_w = self.size  # ASSUME H, W

        if w == target_w and h == target_h:
            return [img, mask]

        if cfg.DATASET.TRANSLATE_AUG_FIX:
            if w < target_w and h < target_h:
                return self.image_in_crop(target_w, target_h, w, h, img, mask)
            else:
                return self.crop_in_image(centroid, target_w, target_h, w, h,
                                          img, mask)

        if self.nopad:
            # Shrink crop size if image < crop
            if target_h > h or target_w > w:
                shorter_side = min(w, h)
                target_h, target_w = shorter_side, shorter_side
        else:
            # Pad image if image < crop
            if target_h > h:
                pad_h = (target_h - h) // 2 + 1
            else:
                pad_h = 0
            if target_w > w:
                pad_w = (target_w - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                mask = ImageOps.expand(mask, border=border,
                                       fill=self.ignore_index)
                w, h = img.size

        return self.crop_in_image(centroid, target_w, target_h, w, h,
                                  img, mask)


class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img, mask):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        return (img.resize((target_w, self.target_h), self.interpolation),
                mask.resize((target_w, self.target_h), Image.NEAREST))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCropPad(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = cfg.DATASET.IGNORE_LABEL

    def __call__(self, img, mask):
        
        assert img.size == mask.size
        w, h = img.size
        if isinstance(self.size, tuple):
                tw, th = self.size[0], self.size[1]
        else:
                th, tw = self.size, self.size
	

        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            # left, top, right, bottom
            img = ImageOps.expand(img, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            mask = ImageOps.expand(mask, border=(pad_x, pad_y, pad_x, pad_y),
                                   fill=self.ignore_index)

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class PadImage(object):
    def __init__(self, size):
        self.size = size
        self.ignore_index = cfg.DATASET.IGNORE_LABEL

        
    def __call__(self, img, mask):
        assert img.size == mask.size
        th, tw = self.size, self.size

        
        w, h = img.size
        
        if w > tw or h > th :
            wpercent = (tw/float(w))    
            target_h = int((float(img.size[1])*float(wpercent)))
            img, mask = img.resize((tw, target_h), Image.BICUBIC), mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        ##Pad
        img = ImageOps.expand(img, border=(0,0,tw-w, th-h), fill=0)
        mask = ImageOps.expand(mask, border=(0,0,tw-w, th-h), fill=self.ignore_index)
        
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(
                Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    """
    Scale image such that longer side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size

        if w > h:
            long_edge = w
        else:
            long_edge = h

        if long_edge == self.size:
            return img, mask

        scale = self.size / long_edge
        target_w = int(w * scale)
        target_h = int(h * scale)
        target_size = (target_w, target_h)

        return img.resize(target_size, Image.BILINEAR), \
            mask.resize(target_size, Image.NEAREST)


class new_Scale(object):
    """
    Scale image such that longer side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)


class ScaleMin(object):
    """
    Scale image such that shorter side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BICUBIC), mask.resize(
                (ow, oh), Image.NEAREST)


class Resize(object):
    """
    Resize image to exact size of crop
    """

    def __init__(self, crop_size):
        self.size = set_crop_size(crop_size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w == h and w == self.size):
            return img, mask
        return (img.resize(self.size, Image.BICUBIC),
                mask.resize(self.size, Image.NEAREST))


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BICUBIC),\
                    mask.resize((self.size, self.size), Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BICUBIC), mask.rotate(
            rotate_degree, Image.NEAREST)


class RandomSizeAndCrop(object):
    def __init__(self, crop_size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, full_size=False,
                 pre_size=None):
        self.crop = RandomCrop(crop_size, nopad=crop_nopad)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.full_size = full_size
        self.pre_size = pre_size

    def __call__(self, img, mask, centroid=None):
        assert img.size == mask.size

        scale_amt = random.uniform(self.scale_min, self.scale_max)

        if self.pre_size is not None:
            in_w, in_h = img.size
            # find long edge
            if in_w > in_h:
                # long is width
                pre_scale = self.pre_size / in_w
            else:
                pre_scale = self.pre_size / in_h
            scale_amt *= pre_scale

        if self.full_size:
            self.crop.size = img.size[1], img.size[0]

        w, h = [int(i * scale_amt) for i in img.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        resized_img, resized_mask = (img.resize((w, h), Image.BICUBIC),
                                     mask.resize((w, h), Image.NEAREST))

        img_mask = self.crop(resized_img, resized_mask, centroid)
        img_mask.append(scale_amt)
        return img_mask


class SlidingCropOld(object):
    def __init__(self, crop_size, stride_rate):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = cfg.DATASET.IGNORE_LABEL

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant',
                      constant_values=self.ignore_label)
        return img, mask

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_sublist, mask_sublist = [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub)
                    img_sublist.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    mask_sublist.append(
                        Image.fromarray(
                            mask_sub.astype(
                                np.uint8)).convert('P'))
            return img_sublist, mask_sublist
        else:
            img, mask = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return img, mask


class SlidingCrop(object):
    def __init__(self, crop_size, stride_rate):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = cfg.DATASET.IGNORE_LABEL

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant',
                      constant_values=self.ignore_label)
        return img, mask, h, w

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_slices, mask_slices, slices_info = [], [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
                    img_slices.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    mask_slices.append(
                        Image.fromarray(
                            mask_sub.astype(
                                np.uint8)).convert('P'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            return img_slices, mask_slices, slices_info
        else:
            img, mask, sub_h, sub_w = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]


class _ClassUniform(object):
    def __init__(self, size, crop_nopad, scale_min=0.5, scale_max=2.0, ignore_index=0,
                 class_list=[16, 15, 14]):
        """
        This is the initialization for class uniform sampling
        :param size: crop size (int)
        :param crop_nopad: Padding or no padding (bool)
        :param scale_min: Minimum Scale (float)
        :param scale_max: Maximum Scale (float)
        :param ignore_index: The index value to ignore in the GT images (unsigned int)
        :param class_list: A list of class to sample around, by default Truck, train, bus
        """
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)

        self.class_list = class_list.replace(" ", "").split(",")

        self.scale_min = scale_min
        self.scale_max = scale_max

    def detect_peaks(self, image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)

        :param image: An 2d input images
        :return: Binary output images of the same size as input with pixel value equal
        to 1 indicating that there is peak at that point
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image == 0)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood,
                                           border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background

        return detected_peaks

    def __call__(self, img, mask):
        """
        :param img: PIL Input Image
        :param mask: PIL Input Mask
        :return: PIL output PIL (mask, crop) of self.crop_size
        """
        assert img.size == mask.size

        scale_amt = random.uniform(self.scale_min, self.scale_max)
        w = int(scale_amt * img.size[0])
        h = int(scale_amt * img.size[1])

        if scale_amt < 1.0:
            img, mask = img.resize((w, h), Image.BICUBIC), mask.resize((w, h),
                                                                       Image.NEAREST)
            return self.crop(img, mask)
        else:
            # Smart Crop ( Class Uniform's ABN)
            origw, origh = mask.size
            img_new, mask_new = \
                img.resize((w, h), Image.BICUBIC), mask.resize((w, h), Image.NEAREST)
            interested_class = self.class_list  # [16, 15, 14]  # Train, Truck, Bus
            data = np.array(mask)
            arr = np.zeros((1024, 2048))
            for class_of_interest in interested_class:
                # hist = np.histogram(data==class_of_interest)
                map = np.where(data == class_of_interest, data, 0)
                map = map.astype('float64') / map.sum() / class_of_interest
                map[np.isnan(map)] = 0
                arr = arr + map

            origarr = arr
            window_size = 250

            # Given a list of classes of interest find the points on the image that are
            # of interest to crop from
            sum_arr = np.zeros((1024, 2048)).astype('float32')
            tmp = np.zeros((1024, 2048)).astype('float32')
            for x in range(0, arr.shape[0] - window_size, window_size):
                for y in range(0, arr.shape[1] - window_size, window_size):
                    sum_arr[int(x + window_size / 2), int(y + window_size / 2)] = origarr[
                        x:x + window_size,
                        y:y + window_size].sum()
                    tmp[x:x + window_size, y:y + window_size] = \
                        origarr[x:x + window_size, y:y + window_size].sum()

            # Scaling Ratios in X and Y for non-uniform images
            ratio = (float(origw) / w, float(origh) / h)
            output = self.detect_peaks(sum_arr)
            coord = (np.column_stack(np.where(output))).tolist()

            # Check if there are any peaks in the images to crop from if not do standard
            # cropping behaviour
            if len(coord) == 0:
                return self.crop(img_new, mask_new)
            else:
                # If peaks are detected, random peak selection followed by peak
                # coordinate scaling to new scaled image and then random
                # cropping around the peak point in the scaled image
                randompick = np.random.randint(len(coord))
                y, x = coord[randompick]
                y, x = int(y * ratio[0]), int(x * ratio[1])
                window_size = window_size * ratio[0]
                cropx = random.uniform(
                    max(0, (x - window_size / 2) - (self.size - window_size)),
                    max((x - window_size / 2), (x - window_size / 2) - (
                        (w - window_size) - x + window_size / 2)))

                cropy = random.uniform(
                    max(0, (y - window_size / 2) - (self.size - window_size)),
                    max((y - window_size / 2), (y - window_size / 2) - (
                        (h - window_size) - y + window_size / 2)))

                return_img = img_new.crop(
                    (cropx, cropy, cropx + self.size, cropy + self.size))
                return_mask = mask_new.crop(
                    (cropx, cropy, cropx + self.size, cropy + self.size))
                return (return_img, return_mask)

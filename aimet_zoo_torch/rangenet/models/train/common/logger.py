# pylint: skip-file

# The MIT License
#
# Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514

import tensorflow as tf
import numpy as np
import scipy.misc
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x


class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tf.summary.FileWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""

    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")

      # Create an Image object
      img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tf.Summary.Value(
          tag='%s/%d' % (tag, i), image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=img_summaries)
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, step)
    self.writer.flush()

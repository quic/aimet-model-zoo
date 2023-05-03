# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

"""module for download progress bar"""
import sys


def printProgressBar(i, max, postText):
    """ print download progress bar"""
    #pylint:disable = redefined-builtin
    n_bar = 10
    j = i / max
    sys.stdout.write("\r")
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}]  {postText}")
    sys.stdout.flush()

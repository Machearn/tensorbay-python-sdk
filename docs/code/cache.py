#!/usr/bin/env python3
#
# Copyright 2021 Graviti. Licensed under MIT License.
#

# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=unsubscriptable-object
# pylint: disable=invalid-name
# pylint: disable=not-an-iterable
# type: ignore[arg-type]

"""This is the example code of using cache for online dataset."""


"""Get Remote Dataset"""
from tensorbay import GAS
from tensorbay.dataset import Dataset

gas = GAS("<YOUR_ACCESSKEY>")
dataset = Dataset("<DATASET_NAME>", gas)
""""""

"""Enable Cache"""
dataset.enable_cache()
""""""

"""Setting Cache Path"""
dataset.enable_cache("<path/to/cache/folder>")
""""""

"""Open Remote Data"""
segment = dataset[0]
MAX_EPOCH = 100
for epoch in range(MAX_EPOCH):
    for data in segment:
        data.open()
        # code using opened data here
""""""

"""Cache Enabled"""
print(dataset.cache_enabled)
# True
""""""

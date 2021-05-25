#!/usr/bin/env python3
#
# Copytright 2020 Graviti. All Rights Reserved
#
# pylint: disable=invalid-name

"""This file defines UAVDT dataloader"""

import os
import re
from collections import defaultdict
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

from ...dataset import Data, Dataset
from ...label import AttributeInfo, Classification, LabeledBox2D
from ...utility import NameOrderedDict
from .._utility import glob

DATASET_NAME = "UAVDT"


def _convert_attr_txt_to_dict(
    path: str, sequence: str, attributes: NameOrderedDict[AttributeInfo]
) -> Dict[str, bool]:
    attr_path = os.path.join(path, "M_attr")
    attr_dict = {}
    with open(os.path.join(attr_path, f"{sequence}_attr.txt")) as file:
        line = file.readline()
        elements = line.strip().split(",")
        attr_dict = {
            attribute_name: bool(int(value)) for attribute_name in attributes for value in elements
        }

    return attr_dict


def _convert_gt_txt_to_dict(
    path: str,
    sequence: str,
    out_of_view_enum: List[str],
    occlusion_enum: List[str],
    catagories: Tuple[str, ...],
) -> Dict[int, List[Dict[str, Any]]]:
    gt_path = os.path.join(path, "UAV-benchmark-MOTD_v1.0", "GT")
    ground_truth_dicts = defaultdict(list)
    with ExitStack() as stack:
        file_mot = stack.enter_context(open(os.path.join(gt_path, f"{sequence}_gt.txt")))
        file_det = stack.enter_context(open(os.path.join(gt_path, f"{sequence}_gt_whole.txt")))
        for line_mot, line_det in zip(file_mot.readlines(), file_det.readlines()):
            ground_truth = {}
            elements_mot = line_mot.strip().split(",")
            elements_det = line_det.strip().split(",")
            ground_truth["target_id"] = elements_det[1]
            ground_truth["bbox_left"] = float(elements_det[2])
            ground_truth["bbox_top"] = float(elements_det[3])
            ground_truth["bbox_width"] = float(elements_det[4])
            ground_truth["bbox_height"] = float(elements_det[5])
            ground_truth["out_of_view"] = out_of_view_enum[int(elements_det[6])]
            ground_truth["occlusion"] = occlusion_enum[int(elements_det[7])]
            ground_truth["object_categories"] = catagories[int(elements_det[8])]
            ground_truth["score"] = float(elements_mot[6])
            frame_id = int(elements_det[0])
            ground_truth_dicts[frame_id].append(ground_truth)
    return ground_truth_dicts


def _convert_dict_to_labels(
    attr_dict: Dict[str, bool], gt_dict_list: Optional[List[Dict[str, Any]]]
) -> Tuple[Classification, List[LabeledBox2D]]:
    classification = Classification(attributes=attr_dict)

    box2d_list = []
    if not gt_dict_list:
        gt_dict_list = []
    for gt_dict in gt_dict_list:
        box2d_attr = {
            "out_of_view": gt_dict["out_of_view"],
            "occlusion": gt_dict["occlusion"],
            "score": gt_dict["score"],
        }
        box2d = LabeledBox2D(
            category=gt_dict["object_categories"],
            attributes=box2d_attr,
            instance=gt_dict["target_id"],
            xmin=gt_dict["bbox_left"],
            ymin=gt_dict["bbox_top"],
            xmax=gt_dict["bbox_left"] + gt_dict["bbox_width"],
            ymax=gt_dict["bbox_top"] + gt_dict["bbox_height"],
        )
        box2d_list.append(box2d)
    return classification, box2d_list


def _extract_frame_id_from_image_path(image_path: str) -> int:
    image_name = os.path.basename(image_path)
    pattern_id = re.compile(r"(?P<id>[1-9]\d*)")
    result = pattern_id.search(image_name)
    return int(result.group("id"))


def UAVDT(path: str) -> Dataset:
    root_path = os.path.abspath(os.path.expanduser(path))

    image_root_path = os.path.join(root_path, "UAV-benchmark-M")
    dataset = Dataset(DATASET_NAME)
    dataset.load_catalog(os.path.join(os.path.dirname(__file__), "catalog.json"))

    box2d_subcatalog = dataset.catalog.box2d
    box_categories = tuple(box2d_subcatalog.categories)
    out_of_view_enum = box2d_subcatalog.attributes["out_of_view"].enum
    occlusion_enum = box2d_subcatalog.attributes["occlusion"].enum

    classfication_attributes = dataset.catalog.classification.attributes

    sequence_list = os.listdir(image_root_path)
    sequence_list.sort()

    for sequence in sequence_list:
        if not os.path.isdir(sequence):
            continue

        segment = dataset.create_segment(sequence)
        sequence_attribute = _convert_attr_txt_to_dict(
            root_path, sequence, classfication_attributes
        )
        ground_truth_dicts = _convert_gt_txt_to_dict(
            root_path, sequence, out_of_view_enum, occlusion_enum, box_categories
        )
        image_path_list = glob(os.path.join(image_root_path, sequence, "*.jpg"))

        for image_path in image_path_list:
            data = Data(image_path)
            frame_id = _extract_frame_id_from_image_path(image_path)
            ground_truth_dict_list = ground_truth_dicts.get(frame_id)
            classification, box2d_list = _convert_dict_to_labels(
                sequence_attribute, ground_truth_dict_list
            )
            data.label.classification = classification
            data.label.box2d = box2d_list
            segment.append(data)

    return dataset

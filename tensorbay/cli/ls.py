#!/usr/bin/env python3
#
# Copyright 2021 Graviti. Licensed under MIT License.
#

"""Implementation of gas ls."""

import sys
from typing import Dict, Iterable, Optional, Union

import click

from ..client import GAS
from ..client.segment import FusionSegmentClient, SegmentClient
from .tbrn import TBRN, TBRNType
from .utility import filter_data, get_dataset_client, get_gas


def _echo_segment(
    dataset_name: str,
    draft_number: Optional[int],
    revision: Optional[str],
    segment: Union[SegmentClient, FusionSegmentClient],
    list_all_files: bool,
) -> None:
    """Echo a segment.

    Arguments:
        dataset_name: The name of the dataset.
        draft_number: The draft number (if the status is draft).
        revision: The revision (if the status is not draft).
        segment: A segment or a fusion segment.
        list_all_files: Only works when segment is a fusion one.
            If False, list frame indexes only.
            If True, list sensors and files, too.

    """
    segment_name = segment.name
    if isinstance(segment, SegmentClient):
        _echo_data(dataset_name, draft_number, revision, segment_name, segment.list_data_paths())
    else:
        frames = segment.list_frames()
        if not list_all_files:
            for index, _ in enumerate(frames):
                click.echo(
                    TBRN(
                        dataset_name,
                        segment_name,
                        index,
                        draft_number=draft_number,
                        revision=revision,
                    ).get_tbrn()
                )
        else:
            for index, frame in enumerate(frames):
                for sensor_name, data in frame.items():
                    click.echo(
                        TBRN(
                            dataset_name,
                            segment_name,
                            index,
                            sensor_name,
                            remote_path=data.path,
                            draft_number=draft_number,
                            revision=revision,
                        )
                    )


def _echo_data(
    dataset_name: str,
    draft_number: Optional[int],
    revision: Optional[str],
    segment_name: str,
    data_iter: Iterable[str],
) -> None:
    """Echo files in data_iter under 'tb:dataset_name:segment_name'.

    Arguments:
        dataset_name: The name of the dataset the segment belongs to.
        draft_number: The draft number (if the status is draft).
        revision: The revision (if the status is not draft).
        segment_name: The name of the segment.
        data_iter: Iterable data to be echoed.

    """
    for data in data_iter:
        click.echo(
            TBRN(
                dataset_name,
                segment_name,
                remote_path=data,
                draft_number=draft_number,
                revision=revision,
            ).get_tbrn()
        )


def _ls_dataset(gas: GAS, info: TBRN, list_all_files: bool) -> None:
    dataset_client = get_dataset_client(gas, info)
    segment_names = dataset_client.list_segment_names()
    if not list_all_files:
        for segment_name in segment_names:
            click.echo(TBRN(info.dataset_name, segment_name).get_tbrn())
        return

    for segment_name in segment_names:
        segment = dataset_client.get_segment(segment_name)
        _echo_segment(info.dataset_name, info.draft_number, info.revision, segment, list_all_files)


def _ls_segment(gas: GAS, info: TBRN, list_all_files: bool) -> None:
    dataset_client = get_dataset_client(gas, info)
    segment = dataset_client.get_segment(info.segment_name)
    _echo_segment(info.dataset_name, info.draft_number, info.revision, segment, list_all_files)


def _ls_frame(gas: GAS, info: TBRN, list_all_files: bool) -> None:
    dataset_client = get_dataset_client(gas, info, is_fusion=True)
    segment_client = dataset_client.get_segment(info.segment_name)

    try:
        frame = segment_client.list_frames()[info.frame_index]
    except IndexError:
        click.echo(f'No such frame: "{info.frame_index}"!', err=True)
        sys.exit(1)

    if not list_all_files:
        for sensor_name in frame:
            click.echo(
                TBRN(
                    info.dataset_name,
                    info.segment_name,
                    info.frame_index,
                    sensor_name,
                    draft_number=info.draft_number,
                    revision=info.revision,
                )
            )
    else:
        for sensor_name, data in frame.items():
            click.echo(
                TBRN(
                    info.dataset_name,
                    info.segment_name,
                    info.frame_index,
                    sensor_name,
                    remote_path=data.path,
                    draft_number=info.draft_number,
                    revision=info.revision,
                )
            )


def _ls_sensor(
    gas: GAS,
    info: TBRN,
    list_all_files: bool,  # pylint: disable=unused-argument
) -> None:
    dataset_client = get_dataset_client(gas, info, is_fusion=True)
    segment_client = dataset_client.get_segment(info.segment_name)
    try:
        frame = segment_client.list_frames()[info.frame_index]
    except IndexError:
        click.echo(f'No such frame: "{info.frame_index}"!', err=True)
        sys.exit(1)

    data = frame[info.sensor_name]
    click.echo(
        TBRN(
            info.dataset_name,
            info.segment_name,
            info.frame_index,
            info.sensor_name,
            remote_path=data.path,
            draft_number=info.draft_number,
            revision=info.revision,
        )
    )


def _ls_fusion_file(
    gas: GAS,
    info: TBRN,
    list_all_files: bool,  # pylint: disable=unused-argument
) -> None:
    dataset_client = get_dataset_client(gas, info, is_fusion=True)
    segment_client = dataset_client.get_segment(info.segment_name)
    try:
        frame = segment_client.list_frames()[info.frame_index]
    except IndexError:
        click.echo(f'No such frame: "{info.frame_index}"!', err=True)
        sys.exit(1)

    if frame[info.sensor_name].path != info.remote_path:
        click.echo(f'No such file: "{info.remote_path}"!', err=True)
        sys.exit(1)

    click.echo(info)


def _ls_normal_file(  # pylint: disable=unused-argument
    gas: GAS, info: TBRN, list_all_files: bool
) -> None:
    dataset_client = get_dataset_client(gas, info, is_fusion=False)
    segment_client = dataset_client.get_segment(info.segment_name)
    _echo_data(
        info.dataset_name,
        info.draft_number,
        info.revision,
        info.segment_name,
        filter_data(segment_client.list_data_paths(), info.remote_path),
    )


_LS_FUNCS = {
    TBRNType.DATASET: _ls_dataset,
    TBRNType.SEGMENT: _ls_segment,
    TBRNType.NORMAL_FILE: _ls_normal_file,
    TBRNType.FRAME: _ls_frame,
    TBRNType.FRAME_SENSOR: _ls_sensor,
    TBRNType.FUSION_FILE: _ls_fusion_file,
}


def _implement_ls(  # pylint: disable=invalid-name
    obj: Dict[str, str], tbrn: str, list_all_files: bool
) -> None:
    gas = get_gas(**obj)

    if not tbrn:
        for dataset_name in gas.list_dataset_names():
            click.echo(TBRN(dataset_name).get_tbrn())
        return

    info = TBRN(tbrn=tbrn)
    _LS_FUNCS[info.type](gas, info, list_all_files)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities for resuming training from checkpoints."""


def resolve_resume_epoch(saved_epoch: int, epoch_finished: bool) -> int:
    """
    Convert the epoch stored in a checkpoint to the epoch to start from when resuming.

    - End-of-epoch checkpoints (epoch_finished=True) store the completed epoch, so
      training continues from the next epoch.
    - Mid-epoch step checkpoints (epoch_finished=False) store the epoch in progress,
      so that epoch is restarted from its beginning.
    """
    if saved_epoch <= 0:
        return 1
    return saved_epoch + 1 if epoch_finished else saved_epoch

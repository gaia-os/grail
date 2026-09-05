"""Thin, cached wrappers around GRAIL Python APIs for the dashboard pages."""

import streamlit as st

from grail.frame import Frame
from grail.frame.registry import FrameRecord, FrameRegistry
from grail.frame.repository import FrameRepository
from grail.frame.state import PosteriorState
from grail.runner.record import RunRecord
from grail.runner.utils import list_all_runs


@st.cache_resource
def get_registry() -> FrameRegistry:
    return FrameRegistry()


@st.cache_resource
def get_repository() -> FrameRepository:
    return FrameRepository()


def list_frame_records() -> list[FrameRecord]:
    return get_registry().list_frames()


def load_frame(spec_path: str) -> Frame:
    return get_repository().load(spec_path)


def get_posterior(frame: Frame, variable_name: str) -> PosteriorState | None:
    return frame.get_posterior(variable_name)


def list_runs() -> list[RunRecord]:
    return list_all_runs()

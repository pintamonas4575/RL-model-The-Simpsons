import streamlit as st
from typing import Any

@st.cache_resource
def _get_storage() -> dict:
    """
    Always returns the same cached dictionary.
    The first time it creates {"my_list": []}, then reuses the same one throughout the session.
    """
    return {"my_list": []}

def cache_gallery_list(lst: list[Any]) -> None:
    """
    Overwrites the entire list stored in the cached dictionary.
    """
    storage = _get_storage()
    storage["my_list"] = lst

def get_cached_gallery() -> list[Any]:
    """
    Returns the complete list saved in cache (or [] if empty).
    """
    return _get_storage()["my_list"]

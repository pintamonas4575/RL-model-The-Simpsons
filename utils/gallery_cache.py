import streamlit as st
from typing import Optional, List, Tuple

# @st.cache_data
# def cache_gallery_images(add: bool = False, img_bytes: Optional[bytes] = None, idx: Optional[int] = None) -> List[Tuple[bytes, int]]:
#     """
#     Manages the cached images list.
#     - If add=True and img_bytes and idx are provided, adds the tuple (img_bytes, idx) to the list.
#     - Always returns the updated list of images [(bytes, index), ...].
#     """
#     # initialize the list only the first time it's called
#     images_list = []

#     # special key _MANAGED in st.session_state for persistance
#     if "_MANAGED_IMAGES_KEY" in st.session_state:
#         images_list = st.session_state["_MANAGED_IMAGES_KEY"]
#     else:
#         st.session_state["_MANAGED_IMAGES_KEY"] = images_list
#     if add and img_bytes is not None and idx is not None:
#         images_list.append((img_bytes, idx))
#         # update list on session_state
#         st.session_state["_MANAGED_IMAGES_KEY"] = images_list

#     return images_list

@st.cache_data
def manage_images(add: bool = False, img_bytes: Optional[bytes] = None, idx: Optional[int] = None) -> List[Tuple[bytes, int]]:
    images_list = []
    if "_MANAGED_IMAGES_KEY" in st.session_state:
        images_list = st.session_state["_MANAGED_IMAGES_KEY"]
    else:
        st.session_state["_MANAGED_IMAGES_KEY"] = images_list
    if add and img_bytes is not None and idx is not None:
        images_list.append((img_bytes, idx))
        st.session_state["_MANAGED_IMAGES_KEY"] = images_list
    return images_list
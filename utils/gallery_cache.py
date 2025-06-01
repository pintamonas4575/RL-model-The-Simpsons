import streamlit as st
from typing import Any, List

@st.cache_resource
def _get_storage() -> dict:
    """
    Devuelve siempre el mismo diccionario cacheado.
    La primera vez se crea {"my_list": []}, y luego se reusa idéntico en toda la sesión.
    """
    return {"my_list": []}

def cache_gallery_list(lst: List[Any]) -> None:
    """
    Sobrescribe la lista entera almacenada en el diccionario cacheado.
    """
    storage = _get_storage()
    storage["my_list"] = lst

def append_to_list(item: Any) -> None:
    """
    Añade un ítem al final de la lista cacheada.
    """
    storage = _get_storage()
    storage["my_list"].append(item)

def get_gallery_list() -> List[Any]:
    """
    Devuelve la lista completa guardada en caché (o [] si está vacía).
    """
    return _get_storage()["my_list"]

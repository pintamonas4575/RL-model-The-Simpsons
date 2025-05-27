import streamlit as st

st.title("Episode Gallery")

images = st.session_state.get("gallery_images", [])
if not images:
    st.info("No hay imágenes en la galería. Genera alguna primero.")
else:
    # Filas de 3 columnas
    cols = st.columns(3)
    for i, img in enumerate(images):
        with cols[i % 3]:
            st.image(img, use_container_width=True, caption=f"Episode {i}")
        if (i+1) % 3 == 0 and i+1 < len(images):
            cols = st.columns(3)

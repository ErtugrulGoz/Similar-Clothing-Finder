import streamlit as st
st.set_page_config(page_title="Similar Clothing Finder", layout="wide")
import pandas as pd
import numpy as np
import torch
import requests
import io
import time
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor
import open_clip
from utils_model import load_selected_model, get_image_embedding, load_image_from_url, MODEL_OPTIONS
from model_comparison_panel import run_comparison_panel

def find_similar_clothes(query_image, image_urls, model, processor, top_n, threshold):
    images = [load_image_from_url(url) for url in image_urls]
    valid_pairs = [(url, img) for url, img in zip(image_urls, images) if img is not None]

    total = len(valid_pairs)
    if total == 0:
        return [], total

    embeddings = []
    progress = st.progress(0, text="🔄 Processing images...")

    for idx, (_, img) in enumerate(valid_pairs):
        emb = get_image_embedding(img, model, processor)
        embeddings.append(emb)
        progress.progress((idx + 1) / total, text=f"{idx + 1}/{total} images processed")

    progress.empty()
    embeddings_matrix = np.array(embeddings)
    query_embedding = get_image_embedding(query_image, model, processor)

    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    filtered_results = []
    for i in sorted_indices:
        score = similarities[i]
        if score >= threshold:
            filtered_results.append({"url": valid_pairs[i][0], "similarity": score})
        if len(filtered_results) == top_n:
            break

    return filtered_results, total

# -------- Streamlit UI --------
st.title("👕 Similar Clothing Finding System")
tab1, tab2 = st.tabs(["🔎 Image Query Panel", "📊 Model Comparison Panel"])

df = pd.read_excel("photodata.xlsx")
df = df[df["ImageURL"].notna()].reset_index(drop=True)

with tab1:
    with st.expander("📂 Image Selection", expanded=False):
        st.markdown("### 🧵 Select Category and Image")

        categories = sorted(df["Category"].dropna().unique())
        selected_category = st.selectbox("🧩 Category", categories)

        category_df = df[df["Category"] == selected_category]


        NUMBER_OF_IMAGES_FOR_PAGE = 50
        sum_img = len(category_df)
        page_number = (sum_img // NUMBER_OF_IMAGES_FOR_PAGE) + 1

        page_no = st.number_input(f"📄 Page", min_value=1, max_value=page_number, value=1, step=1)
        st.markdown(f"{page_no} / {page_number}")
        starting_index = (page_no - 1) * NUMBER_OF_IMAGES_FOR_PAGE
        ending_index = min(starting_index + NUMBER_OF_IMAGES_FOR_PAGE, sum_img)

        img_cols = st.columns(5)
        for local_idx in range(starting_index, ending_index):
            row = category_df.iloc[local_idx]
            url = row["ImageURL"]
            img = load_image_from_url(url)
            if img:
                with img_cols[local_idx % 5]:
                    st.image(img.resize((150, 150)), use_container_width=True)
                    if st.button("Query this image", key=f"select_{row.name}"):
                        st.session_state["query_index"] = row.name
                        st.rerun()

    st.markdown("### ⚙️ Search Settings")
    model_key = st.selectbox("📦 Model Selection", list(MODEL_OPTIONS.keys()))
    top_n = st.slider("📊 Number of Most Similar to Display", 1, 50, 5)
    similarity_threshold = st.slider("🎯 Minimum Similarity Threshold", 0.0, 1.0, 0.5, 0.01)

    if "query_index" in st.session_state:
        selected_row = df.loc[st.session_state["query_index"]]
        query_url = selected_row["ImageURL"]
        query_image = load_image_from_url(query_url)
        if query_image is None:
            st.error("❌ The selected image URL could not be loaded. Please choose a different image.")
            st.stop()

        st.markdown("### 📸 Selected Image and Automatic Filters")
        st.image(query_image, caption="Query Image", width=300)
        st.write(f"- **ID**: {selected_row['ID']}")
        st.write(f"- **Gender**: {selected_row['Gender']}")
        st.write(f"- **Category**: {selected_row['Category']}")
        st.write(f"- **Main Color**: {selected_row['MainColor']}")

        filtered_df = df[
            (df["Gender"] == selected_row["Gender"]) &
            (df["Category"] == selected_row["Category"]) &
            (df["MainColor"] == selected_row["MainColor"]) &
            (df["ImageURL"] != query_url)
        ]
        image_urls = filtered_df["ImageURL"].dropna().tolist()

        if st.button("🔎 Find Similar"):
            if not image_urls:
                st.warning("⚠️ No other images found for the selected category.")
            else:
                model, processor = load_selected_model(model_key)
                start_time = time.time()

                results, total_checked = find_similar_clothes(
                    query_image, image_urls, model, processor, top_n, similarity_threshold
                )
                elapsed = time.time() - start_time

                st.success(f"✅ Found {len(results)} similar items among {total_checked} images ({elapsed:.2f} seconds)")

                if results:
                    st.subheader("📸 Most Similar Clothes")
                    result_cols = st.columns(5)
                    for idx, result in enumerate(results):
                        img = load_image_from_url(result["url"])
                        if img:
                            with result_cols[idx % 5]:
                                st.image(img.resize((200, 200)), use_container_width=True)
                                st.caption(f"Score: {result['similarity']:.2f}")
                else:
                    st.warning("❌ No similar images matching the threshold were found. You can choose a different image and try again.")
with tab2:
    if "query_index" in st.session_state:
        selected_row = df.loc[st.session_state["query_index"]]
        query_image = load_image_from_url(selected_row["ImageURL"])
        filtered_df = df[
            (df["Gender"] == selected_row["Gender"]) &
            (df["Category"] == selected_row["Category"]) &
            (df["MainColor"] == selected_row["MainColor"]) &
            (df["ImageURL"] != selected_row["ImageURL"])
        ]
        image_urls = filtered_df["ImageURL"].dropna().tolist()

        if query_image and image_urls:
            run_comparison_panel(query_image, image_urls, top_n=10, query_row=selected_row, df=df)



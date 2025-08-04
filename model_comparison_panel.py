import time
import requests
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils_model import load_selected_model, get_image_embedding, load_image_from_url, MODEL_OPTIONS
import pandas as pd
from matplotlib.cm import get_cmap

df = pd.read_excel("photodata.xlsx") 

def run_comparison_panel(query_image, image_urls, top_n=10, query_row=None, df=None):
    st.header("📊 Model Comparison Panel")
    model_results = {}
    timing_results = {}

    for model_key in MODEL_OPTIONS:
        model, processor = load_selected_model(model_key)
        start = time.time()
        results = []
        for url in image_urls:
            img = load_image_from_url(url)
            if img is None:
                continue
            emb = get_image_embedding(img, model, processor)
            query_emb = get_image_embedding(query_image, model, processor)
            sim = cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1))[0][0]
            results.append({"url": url, "similarity": sim})
        sorted_res = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]
        model_results[model_key] = sorted_res
        timing_results[model_key] = time.time() - start
        

    st.subheader("🔍 Query Image")
    st.image(query_image, caption="Query Image", width=300)


    query_url = None
    for url in image_urls:
        if url not in [r["url"] for rs in model_results.values() for r in rs]:
            query_url = url
            break

    query_row = df[df["ImageURL"] == query_url].iloc[0] if query_url else None
    if query_row is not None:
        st.write(f"- **ID**: {query_row['ID']}")
        st.write(f"- **Gender**: {query_row['Gender']}")
        st.write(f"- **Category**: {query_row['Category']}")
        st.write(f"- **Main Color**: {query_row['MainColor']}")

    st.subheader("📈 Model × Similarity Scores")

    df = pd.read_excel("photodata.xlsx")  
    urun_ref_set = set()
    for results in model_results.values():
        for r in results:
            matched = df[df["ImageURL"] == r["url"]]
            if not matched.empty:
                urun_ref_set.add(matched["ID"].values[0])

    urun_ref_list = sorted(list(urun_ref_set))
    color_map = get_cmap("tab20")
    ref_to_color = {ref: color_map(i / len(urun_ref_list)) for i, ref in enumerate(urun_ref_list)}

    plt.figure(figsize=(12, 6))

    for model_key, results in model_results.items():
        x_vals = [model_key] * len(results)
        y_vals = []
        colors = []
        for r in results:
            matched = df[df["ImageURL"] == r["url"]]
            if matched.empty:
                continue
            ref = matched["ID"].values[0]
            score = r["similarity"]
            y_vals.append(score)
            colors.append(ref_to_color[ref])
            plt.scatter(model_key, score, color=ref_to_color[ref], label=ref)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", title="ID")

    plt.ylabel("Similarity Score")
    plt.xlabel("Model")
    plt.title("The Products Found Most Similar by Models (ID) and Their Scores")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("📈 Score Comparison")
    plt.figure(figsize=(10, 5))
    for model_key, results in model_results.items():
        scores = [r["similarity"] for r in results]
        plt.plot(range(len(scores)), scores, label=model_key, marker="o")
    plt.xlabel("Order")
    plt.ylabel("Similarity Score")
    plt.title("Model Score Distribution")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader("⏱️ Processing Times")
    for model_key, t in timing_results.items():
        st.write(f"- {model_key}: {t:.2f} saniye")

    st.subheader("🖼️ Top-10 Images")

    for model_key, results in model_results.items():
            st.markdown(f"### 🔹 {model_key}")
            cols = st.columns(5)
            for i, r in enumerate(results):
                img = load_image_from_url(r["url"])
                if img:
                    with cols[i % 5]:
                        st.image(img.resize((200, 200)), use_container_width=True)
                        ref = df[df["ImageURL"] == r["url"]]["ID"].values
                        urun_ref = ref[0] if len(ref) > 0 else "Uncertain"
                        st.caption(f"Score: {r['similarity']:.2f}\nRef: {id}")

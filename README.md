# Similar Clothing Finder

## Introduction

This project is a web application built with Streamlit that allows users to find similar clothing items based on a query image. It provides a user-friendly interface to browse a catalog of clothes, select an item, and find the most similar products.

A key feature of this application is the "Model Comparison Panel," which enables users to evaluate the performance of different pre-trained deep learning models (CLIP and SigLIP based models) for the similarity search task. It provides a side-by-side comparison of the results, scores, and processing times for each model.  The application utilizes five distinct zero-shot models for similarity search.



## Features

* **Image Query Panel:**
    * Browse clothing items by category.
    * Select an image from the catalog to use as a query.
    * Find the most similar items from a filtered dataset (based on gender, category, and main color).
    * Customize search parameters like the number of similar items to display and a minimum similarity threshold.
* **Model Comparison Panel:**
    * Compare multiple deep learning models' performance on the same query image.
    * Visualize the similarity scores of the found products for each model using scatter plots and line charts.
    * Display the top-10 similar images found by each model.
    * Show the processing time for each model to evaluate efficiency.
      
The code in this repository has been prepared for public use and differs from the original project developed. 

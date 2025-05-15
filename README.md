# Image Compression using Singular Value Decomposition (SVD)

## Overview: The Science Behind the Project
This project leverages Singular Value Decomposition (SVD), a powerful linear algebra technique, to perform image compression. The idea is based on the fundamental insight that not all information in an image is equally important. By decomposing the image into its singular values and vectors, we can selectively retain only the most significant components, effectively compressing the image while preserving its core structure.

Key Idea: SVD captures the most meaningful structure in a matrix and discards the noise or less significant information by keeping only the top singular values. This technique is also the backbone of Principal Component Analysis (PCA) in data science.

## Scientific Concepts

**SVD** is a matrix factorization method that decomposes any real or complex matrix \( A \) into three matrices:

**A = U Σ Vᵗ**


Where:
- `U` is an orthogonal matrix whose columns are the left singular vectors.
- `Σ` is a diagonal matrix containing the singular values in descending order.
- `Vᵗ` is the transpose of an orthogonal matrix whose rows are the right singular vectors.

In the context of image processing, images are treated as matrices (for grayscale) or 3D tensors (for RGB images). Each color channel (Red, Green, Blue) is processed separately using SVD.

The idea is to keep only the top **k** singular values, which hold the most significant information about the image. The rest, which often correspond to noise or minor details, are discarded. This yields a lower-rank approximation of the image that uses less storage.

## Applications of SVD

- **Image Compression:** Reduce image size by approximating each color channel using a lower-rank matrix.
- **Noise Reduction:** Discarding small singular values effectively removes minor variations and noise.
- **Dimensionality Reduction:** SVD is at the core of **Principal Component Analysis (PCA)**, which helps in visualizing high-dimensional data in 2D or 3D.

## How It Works in This Project

1. The uploaded or captured image is converted into its RGB components.
2. SVD is applied independently to each color channel.
3. Each channel is reconstructed using only the top **k** singular values.
4. The compressed channels are stacked together and normalized to form the final image.
5. The compressed image can be displayed and downloaded in JPEG format.

## Usage Interface

This project provides a **Streamlit** web application that allows:

- Uploading an image from disk or capturing it using a webcam.
- Selecting the number of singular values (**k**) using a slider.
- Displaying the original and compressed images side by side.
- Downloading the compressed image.

## Benefits of SVD-based Compression

- **Adjustable Quality:** The slider for **k** allows users to control the trade-off between image quality and compression.
- **Educational Value:** Visualizing the impact of matrix rank on image quality is a great learning tool for linear algebra and data science.
- **Versatility:** The same principles are used in many fields, including signal processing, natural language processing, and recommender systems.

## Conclusion

This project illustrates how mathematical concepts like SVD can be applied in real-world tasks like image compression. It bridges the gap between theoretical linear algebra and practical applications, and provides a flexible tool to understand and visualize data in reduced dimensions.

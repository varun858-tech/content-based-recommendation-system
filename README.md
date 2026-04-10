# Content-Based Movie Recommendation System 🎬

## Level 1: The Baseline Model (Binary Feature Matrix)
This repository contains the first iteration of a content-based recommendation engine. The objective of this phase was to construct a foundational system relying strictly on movie genres using a **Binary Feature Matrix Vectorizer**.

### The Methodology
1. **Data Ingestion & Cleaning:** Processed the `movies.dat` file, parsing the double-colon (`::`) separators and normalizing the text by converting all genres to lowercase.
2. **Feature Engineering:** Transformed the pipe-separated string of genres into a binary matrix using Scikit-Learn's `MultiLabelBinarizer`. This creates a mathematical vocabulary where each movie is represented as a vector of `0`s and `1`s.
3. **Mathematical Core:** The system calculates the proximity between movie vectors by computing the **Cosine Similarity** of the $(10k \times 10k)$ matrix:
   $$Similarity = \frac{A \cdot B}{|A| \cdot |B|}$$
4. **Retrieval:** Given an input title, the system isolates the movie's index, sorts the similarity distribution, and returns the top 5 closest matches.

### Findings & System Limitations
While this baseline functions correctly mechanically, a manual review of the output revealed several critical flaws that require addressing in subsequent levels:
* **Over-Reliance on Genre:** The model uses only one type of feature, leading to extremely low feature complexity.
* **Bucket Collapsing:** Because broad categories like "Drama" or "Action" are treated as coarse features, vastly different movies collapse into the same recommendation bucket. 
* **Zero Thematic Understanding:** The model lacks the semantics to distinguish between tones or emotions. It cannot differentiate between themes like "hope," "despair," or "rebellion." For example, *Fight Club* is treated identically to any standard action movie. 
* **Equal Weighting:** Every genre is weighted equally ($Drama = Film-Noir = Sci-fi$), allowing highly common genres to heavily dominate the similarity scores.
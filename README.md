# Content-Based Movie Recommendation System 🎬

## Level 1: The Baseline Model (Binary Feature Matrix)
This repository contains the first iteration of a content-based recommendation engine. [cite_start]The objective of this phase was to construct a foundational system relying strictly on movie genres using a **Binary Feature Matrix Vectorizer**[cite: 1553].

### The Methodology
1. [cite_start]**Data Ingestion & Cleaning:** Processed the `movies.dat` file, parsing the double-colon (`::`) separators and normalizing the text by converting all genres to lowercase [cite: 1694-1695, 1728].
2. [cite_start]**Feature Engineering:** Transformed the pipe-separated string of genres into a binary matrix using Scikit-Learn's `MultiLabelBinarizer` [cite: 1746-1748, 1759-1760]. [cite_start]This creates a mathematical vocabulary where each movie is represented as a vector of `0`s and `1`s[cite: 1568, 1570].
3. [cite_start]**Mathematical Core:** The system calculates the proximity between movie vectors by computing the **Cosine Similarity** of the $(10k \times 10k)$ matrix [cite: 1826-1831]:
   $$Similarity = \frac{A \cdot B}{|A| [cite_start]\cdot |B|}$$ [cite: 1580]
4. [cite_start]**Retrieval:** Given an input title, the system isolates the movie's index, sorts the similarity distribution, and returns the top 5 closest matches [cite: 1954-1979].

### Findings & System Limitations
While this baseline functions correctly mechanically, a manual review of the output revealed several critical flaws that require addressing in subsequent levels:
* [cite_start]**Over-Reliance on Genre:** The model uses only one type of feature, leading to extremely low feature complexity [cite: 2036-2039, 2075-2081].
* [cite_start]**Bucket Collapsing:** Because broad categories like "Drama" or "Action" are treated as coarse features, vastly different movies collapse into the same recommendation bucket [cite: 2049-2060]. 
* **Zero Thematic Understanding:** The model lacks the semantics to distinguish between tones or emotions. [cite_start]It cannot differentiate between themes like "hope," "despair," or "rebellion" [cite: 2061-2065]. [cite_start]For example, *Fight Club* is treated identically to any standard action movie[cite: 2067]. 
* [cite_start]**Equal Weighting:** Every genre is weighted equally ($Drama = Film-Noir = Sci-fi$), allowing highly common genres to heavily dominate the similarity scores [cite: 2068-2074].
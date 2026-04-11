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


## Level 2: Scaling & Feature Fusion (Bag of Words)
This repository contains the second iteration of the recommendation engine. The objective of this phase was to transition from a rigid, genre-only model to a system capable of capturing narrative themes and tones using a **Bag of Words (BoW)** approach on a significantly larger dataset.

### The Methodology
1. **Data Ingestion & Merging:** Transitioned to the MovieLens dataset, managing over 62,000 films. Ingested and merged the movies and tags datasets, systematically handling null values and cleaning the text.
2. **Feature Engineering (The "Bag"):** Fused movie genres and user-generated tags into a single, lowercase string per movie. Transformed this text using Scikit-Learn's `CountVectorizer` to build a vocabulary of the **5,000 most frequent words**, actively filtering out unhelpful English "stop words."
3. **Mathematical Core & Optimization:** The system calculates **Cosine Similarity** based on word frequency rather than binary presence. To prevent CPU Out-Of-Memory (OOM) crashes from computing a massive $(62k \times 62k)$ matrix, the architecture was optimized to calculate similarity **"on the go"**—comparing only the user's requested movie vector against the rest of the dataset.
4. **Retrieval:** Given an input string, the system performs a lowercase subset match to isolate the requested movie, computes the on-the-go similarity distribution, and returns the top 5 closest thematic matches.

### Findings & System Limitations
This model represents a massive upgrade in tonal recognition. For example, *Fight Club* successfully groups with *The Machinist* and *Shutter Island* because the model identifies multi-dimensional tags like "psychological," "dark," and "identity crisis." However, significant flaws remain:
* **Zero Semantic Understanding:** The model only counts word occurrences; it has no true comprehension of meaning. It cannot distinguish between conceptually opposite words like "hope" and "despair."
* **Frequency Bias:** Highly common descriptive words (e.g., "drama" or "crime") heavily dominate the mathematical similarity scores, often drowning out rare but highly specific and powerful thematic keywords.
* **Lack of Hierarchy:** Every word is weighted identically ("dark" == "prison" == "based on novel"), which does not align with how humans actually prioritize concepts when recommending a film.
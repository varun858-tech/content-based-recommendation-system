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

## Level 3: Word Normalization (Stemming & Lemmatization)
This repository contains the third iteration of the recommendation engine. The objective of this phase was to implement Natural Language Processing (NLP) techniques to clean the "noise" in the Bag of Words vocabulary, ensuring that different tenses of the same word (e.g., "loves", "loving", "loved") are mathematically treated as the same feature.

### The Methodology
1. **The Stemming Approach:** Initially applied NLTK's `PorterStemmer` to forcefully chop words down to their root forms.
2. **The Lemmatization Upgrade:** Recognizing the flaws in blunt stemming, the pipeline was upgraded to use `WordNetLemmatizer`.
3. **Part-of-Speech (POS) Tagging:** To make lemmatization accurate, a custom function was engineered to tag the grammatical role of every word. This ensures context is preserved (e.g., "running" as a verb becomes "run", but "running" as a noun stays "running").
4. **Re-Vectorization:** The cleaned "bag" was passed back through the `CountVectorizer` and the on-the-go Cosine Similarity logic to generate new recommendations.

### Findings & System Limitations
* **The Stemming Flaw (Accidental Overlap):** Stemming proved to be too aggressive and context-blind. For example, "universe" (space film) and "university" (college film) were both stemmed to the root `"univers"`. This caused the model to falsely assume high similarity between completely unrelated movies. 
* **The Lemmatization Trade-Off:** While Lemmatization with POS tagging completely fixed the accidental overlaps, it was significantly slower and more computationally expensive to run across a dataset of 62,000 films.
* **The Ultimate Reality Check:** The most crucial finding of Level 3 is that word normalization ultimately had **only a minor impact** on the final recommendations. While the vocabulary was cleaner, the core mathematical logic remained unchanged: the system still only *counts* words. It still lacks true semantic understanding and still suffers from frequency bias.

## Level 4: Thematic Weighting (TF-IDF)
This repository contains the final iteration of the recommendation engine. The objective of this phase was to solve the "Frequency Bias" inherent in the Bag of Words model by implementing **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization.

### The Methodology
1. **Mathematical Weighting:** Replaced `CountVectorizer` with Scikit-Learn's `TfidfVectorizer`. Instead of merely counting words, the system now calculates a weight for each word based on two factors:
   * **Term Frequency (TF):** How often a word appears in a specific movie's feature set.
   * **Inverse Document Frequency (IDF):** How rare the word is across the entire 62,000+ movie dataset.
2. **Algorithmic Shift:** Common genre tags (like "action" or "sci-fi") mathematically receive a massive penalty, while rare, highly specific thematic tags (like "heist" or "time-loop") receive a significant boost.
3. **Similarity Calculation:** The system continues to use on-the-go **Cosine Similarity**, but the geometric distance is now dictated by shared *important* themes rather than shared *common* words.

### Findings & System Limitations
The implementation of TF-IDF successfully transformed the recommendation logic. For example, when searching for *Inception*, the model accurately down-weighted generic overlaps like "sci-fi" causing previously highly-ranked films to drop, prioritizing films with higher thematic density instead. However, the foundational limits of keyword matching remain:
* **Zero Semantics:** The model still lacks true semantic comprehension. It cannot recognize that "dream" and "subconscious" mean the same thing.
* **Exact Word Dependency:** If two movies express the exact same concept using different vocabulary, the system completely misses the connection.
* **Context Blindness:** The mathematical weighting does not understand the context or grammatical role of the words being used.
* **Vector Sparsity:** The mathematical vectors remain highly dimensional and are mostly populated by zeroes, which leaves room for future optimization.

## Level 5: The Hybrid Discovery Pipeline (Production-Ready)
This repository contains the final, production-ready iteration of the recommendation engine. The objective of this phase was to transition from an academic Jupyter Notebook environment into a robust Python pipeline, integrating custom feature weighting and Bayesian statistics to deliver high-quality, thematic recommendations.

### The Methodology
1. **Bayesian Weighted Ratings:** Implemented the IMDB weighted rating formula: $\frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C$. This prevents obscure films with a single 10/10 rating from outranking critically acclaimed classics with thousands of reviews.
2. **Custom Feature Weighting:** Engineered the tags by mathematically boosting the director (`crew * 3`) and thematic elements (`keywords * 4`, `genres * 4`). This forces the algorithm to prioritize the "creative DNA" of a film over generic plot summaries.
3. **Hybrid Scoring Algorithm:** Calculated a final recommendation score blending three distinct metrics:
   * **60%** Content Similarity (TF-IDF Cosine Distance)
   * **25%** Bayesian Rating (Quality)
   * **15%** Popularity (Normalized Vote Count)
4. **Director's Spotlight:** Engineered a secondary retrieval function to surface other top-rated films by the target movie's director, enhancing the user discovery experience.

### Findings & System Limitations
* **High Thematic Accuracy:** The hybrid approach successfully identifies complex cinematic styles. For example, it accurately matched *Pather Panchali* with *Bicycle Thieves*, recognizing the underlying "Neorealism" tone across different decades and languages.
* **Quality Control:** The integration of the Bayesian rating successfully filters out low-quality "trash" matches. The system no longer recommends objectively bad films just because they share the same keywords.
* **The Serialization Bottleneck:** While the pipeline produces A-tier results, running the full NLP lemmatization and TF-IDF vectorization on 62,000+ movies on-the-fly is computationally expensive. For a web deployment (e.g., Streamlit), the processed dataframes and similarity vectors must be serialized (Pickled) to avoid massive loading times.
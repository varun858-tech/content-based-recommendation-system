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

## Level 5.1: Metadata Fusion & Hybrid Ranking

This repository contains the fifth iteration of the recommendation engine. The objective of this phase was to move beyond user-generated tags and build a richer representation of a film by integrating multiple metadata sources. The goal was to capture the complete "creative identity" of a movie through its cast, director, keywords, genres, and plot description while simultaneously improving recommendation quality through ranking heuristics.

### The Methodology

1. **Dataset Expansion:** Transitioned from the MovieLens dataset to the TMDB Metadata Dataset, merging information from `movies_metadata.csv`, `credits.csv`, and `keywords.csv`.
2. **Metadata Extraction:** Parsed JSON-like structures to extract:

   * Genres
   * Top 3 Cast Members
   * Director
   * Keywords
   * Plot Overview
3. **Custom Feature Weighting:** Engineered a weighted metadata representation:

   * Genres × 4
   * Keywords × 4
   * Director × 3
   * Cast × 2
   * Overview × 1

   This weighting strategy forces the model to prioritize the creative DNA of a film over generic descriptive text.
4. **Advanced Vectorization:** Applied Lemmatization followed by TF-IDF Vectorization with `ngram_range=(1,3)` to capture both individual words and meaningful phrases.
5. **Quality Re-Ranking:** Instead of relying solely on cosine similarity, a final recommendation score was computed using:

   * **60%** Content Similarity
   * **25%** Rating Score
   * **15%** Popularity Score
6. **Director's Spotlight:** Implemented a secondary recommendation module that surfaces other highly-rated works from the target film's director.

### Findings & System Limitations

This iteration produced the first genuinely high-quality recommendations in the project lifecycle. For example:

* *Fight Club* successfully surfaced films such as *Se7en*, *Oldboy*, and *The Machinist*, reflecting psychological themes rather than merely genre overlap.
* *Pather Panchali* aligned with films such as *Bicycle Thieves* and *The World of Apu*, indicating that the system was beginning to capture cinematic style and narrative tone.
* The Director's Spotlight feature significantly improved user discovery and exploration.

However, several limitations remained:

* **Manual Feature Engineering:** The weighting coefficients (`genres × 4`, `keywords × 4`, etc.) were selected heuristically rather than learned automatically.
* **Vocabulary Dependency:** The model still relies on exact keyword overlap and cannot recognize semantic equivalents.
* **Sparse Representations:** Despite TF-IDF improvements, the vectors remain highly sparse and computationally expensive.

---

## Level 5.2: NLP Refinement & Production Optimization

This repository contains the sixth iteration of the recommendation engine. The objective of this phase was not to redesign the recommendation algorithm, but rather to improve the quality of the textual feature space by reducing linguistic noise and strengthening the influence of meaningful narrative information.

### The Methodology

1. **Overview Cleaning:** Implemented NLTK English Stopword Removal to eliminate low-information words such as:

   * "the"
   * "is"
   * "was"
   * "and"
   * "of"
2. **Narrative Compression:** Limited plot summaries to the first 50 meaningful words after stopword filtering, preventing excessively long overviews from dominating the feature space.
3. **Metadata Preservation:** Retained the weighted metadata architecture introduced in Level 5.1:

   * Genres × 4
   * Keywords × 4
   * Director × 3
   * Cast × 2
   * Overview × 1
4. **Re-Vectorization:** Rebuilt TF-IDF vectors using the cleaner feature representation and regenerated recommendation rankings using the hybrid scoring pipeline.

### Findings & System Limitations

This phase produced smaller but important improvements:

* **Cleaner Narrative Signals:** Plot summaries now contribute meaningful concepts rather than grammatical filler words.
* **Improved Stability:** Recommendations became less sensitive to noisy wording inside movie overviews.
* **Better Feature Utilization:** The weighted metadata components exert greater influence because they are no longer competing with large volumes of low-information text.

The recommendation outputs remained largely consistent with Level 5.1, which is a positive indicator that the model's thematic understanding was already robust.

However, the fundamental limitations remain unchanged:

* **No Semantic Understanding:** The model still cannot understand that words such as "automobile" and "car" refer to the same concept.
* **No Contextual Reasoning:** TF-IDF remains a keyword-based system and cannot infer meaning from sentence structure.
* **Feature Weights Remain Handcrafted:** The ranking mechanism still depends on manually engineered coefficients.
* **Classical NLP Ceiling:** Further preprocessing improvements now yield diminishing returns.

### Key Takeaway

Level 5.2 represents the practical ceiling of a classical TF-IDF recommendation architecture. At this stage, the system combines:

* Metadata Fusion
* Feature Weighting
* Lemmatization
* Stopword Removal
* TF-IDF Vectorization
* Hybrid Ranking
* Director-Based Discovery

Future improvements require a fundamental shift away from keyword matching and toward semantic embeddings generated by modern transformer-based language models.

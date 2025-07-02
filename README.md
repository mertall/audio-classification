# Water Bottle Challenge

**Purpose**: Build an unsupervised classifier to distinguish between knife-strikes on the *top* vs *bottom* of a stainless steel water bottle, using only two labeled examples and 24 unlabeled spectrogram CSVs.    
**Results** 95% accuracy on labeled data!
---

## ✅ Project Overview
Most code in this challenge was developed with help of ChatGPT. I gave it a set of requirements for my code and had it create the scaffolding. 
Given 3 hour limit, I wanted to honor it as best I could, and using ChatGPT helped me iterate quickly.
My exploration.ipynb took about 2 hours of work. 
The actual water_bottle_challenge.py, the tests, and readme were created in just over 1 hour - this required more manual coding versus ChatGPT assist. 
1. **Data**  
   - Preprocessed spectrogram CSVs in `data/`:  
     - `top.csv`, `bottom.csv` (one clear example each).  
     - `unlabeled_XX.csv` files (unknown strike location).  
   - Each CSV: rows = frequencies (Hz), columns = timepoints (ms), values = magnitude.

2. **Goal**  
   - Implement `classify_preprocessed_audio(fpath: str) -> int`:  
     - **0** = top strike  
     - **1** = bottom strike  
     - **None** = unsure or invalid  
   - Must not error on any valid CSV.

3. **Constraints**  
   - Only two labeled samples.  
   - No external labels for evaluation.  
   - Code readability, concise explanations, and reproducibility matter.

---

## 🛠 Steps & Rationale

### 1. Initial Feature Engineering  
These features were created with help from ChatGPT, I used it to research typical features in audio analysis of spectograms
- **Spectral Centroid**: average frequency weighted by energy → indicates brightness.  
- **Spectral Bandwidth**: RMS spread around centroid → harmonic richness.  
- **Rolloff 85%**: cutoff where 85% of energy resides → high-frequency content.  
- **Peak Frequency**: single frequency with maximum total energy.  

*Rationale*: Top strikes produce sharper, tinny high-freq bursts; bottom strikes ring with lower, sustained tones as tested on a waterbottle in my room!

### 2. Visual Inspection  
- Plotted log-scaled spectrograms with `librosa.display.specshow`.  
- Noted distinct color-dominant zones for top vs bottom.
- Tried to just seperate by where the longest and most colorful streak was

### 3. Cosine angle similarity Prototyping  
- Take cosine sim which is inverse cosine of (dot product of vectors divided by their (magnitudes multiplied))
- Only looks at angle between vectors, which loses magnitude information
- With this data I believe it is not appropriate discard magnitude of the vector when classifying, magnitude should be important in classifying top and bottom
- Though I did utilize these results to validate if K-means was at least on track, and found about 70% agreement between this method and K-means by hand
- Visually looking at cosine results after labeling I felt there were different labels that were wrong after PCA was applied.

### 4. K-means
- **Extract** features for all files.  
- **Standardize** via `StandardScaler`.  
- **KMeans** with **3 clusters** (top, bottom, unsure).  
- **Map clusters** to labels via `top.csv`/`bottom.csv`.  
- **Predict** new files’ cluster and translate to 0/1/None.

*Why 3 clusters?* Some files are ambiguous—allow an “unsure” group. K-means always makes a best guess, so we don't want it guessing top or bottom when really it should be "unsure".

### 5. Feature Expansion & Evaluation  
- Added extras: spectral flatness, time-to-peak, decay rate, low/high energy ratio, spectral entropy.  
- Automated **silhouette score** evaluation.  
- Visualized PCA+KMeans to inspect separation.
- These features were derived with help from ChatGPT, I wanted to quickly evaluate if there were other features I could test and see if it helped our groupings

### 6. Unit Testing & Robustness  
- Created pytest tests verifying `top.csv → 0`, `bottom.csv → 1`, and unlabeled handling.  
---

## ▶️ How to Run

1. **Install**:  
    ```
    pip install -r requirements.txt
    ```
2. **Test**:
    ```
    pytest -v -s test_water_bottle_challenge.py
    ```
3. **Run**:
    ```
    from water_bottle_challenge import classify_preprocessed_audio
    print(classify_preprocessed_audio('data/unlabeled_00.csv'))
    ```

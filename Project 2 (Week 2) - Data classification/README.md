# Project 2 — Data Classification Using AI
DecodeLabs AI Internship | Batch 2026

Predicts **OrderStatus** (Cancelled / Delivered / Pending / Returned / Shipped) for
e-commerce orders using a **K-Nearest Neighbors (KNN)** classifier, following the
Input → Process → Output pipeline from the project brief.

## Files
- `project2_classification.py` — the full pipeline (load → engineer features → split →
  scale → tune K → train → evaluate)
- `Dataset_for_Data_Analytics_-_Sheet1.csv` — the dataset (1200 orders)
- `requirements.txt` — Python dependencies

## How to run in VS Code

1. **Open the folder** in VS Code (`File > Open Folder`), making sure
   `project2_classification.py` and the `.csv` file sit in the same folder.

2. **Create a virtual environment** (recommended) — open a terminal in VS Code
   (`` Ctrl+` ``) and run:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script**:
   ```bash
   python project2_classification.py
   ```
   (Or click the ▶ "Run Python File" button in the top-right of VS Code.)

5. **Check the results**:
   - Console output shows dataset stats, feature list, best K, accuracy, F1 score,
     and a full classification report.
   - An `outputs/` folder is created automatically with:
     - `elbow_curve.png` — error rate vs. K, used to pick the best K
     - `confusion_matrix.png` — visual breakdown of correct vs. incorrect predictions

## About the results (important for your report)

Unlike the Iris dataset shown in the slides — where petal/sepal measurements are strongly
tied to species — `OrderStatus` in this dataset is **not strongly predictable** from the
other columns (Quantity, UnitPrice, Product, PaymentMethod, etc.). You'll see accuracy land
around 20–25%, only slightly above the 20% random-guess baseline for 5 balanced classes.

**This is not a bug — it's a legitimate and useful finding.** It directly demonstrates the
"Output Validation / Accuracy Mirage" concept from your slide deck: a model's accuracy
number only means something once you check it against a sensible baseline (here, random
guessing across 5 classes = ~20%) and inspect the confusion matrix rather than trusting a
single headline metric. For your report, you can honestly state:
- The pipeline (load → split → scale → KNN → evaluate) is correctly implemented and
  produces training/testing sets, a tuned K via the elbow method, and full evaluation
  metrics (accuracy, precision, recall, F1, confusion matrix).
- The low accuracy reflects the dataset itself: `OrderStatus` appears to be assigned
  independently of the other fields (i.e., there's little real signal for KNN to learn),
  which is common in synthetic/simulated business datasets.
- This is exactly why the confusion matrix and F1 score matter more than a raw accuracy
  score — they reveal whether the model is doing better than chance and where it's
  making the most mistakes.

If you want a higher-accuracy result for the demo/portfolio angle instead, the same script
can be pointed at the classic Iris dataset (`sklearn.datasets.load_iris`) — let me know and
I can add that as an alternate mode.

## Customizing
- Change `TEST_SIZE`, `RANDOM_STATE`, or `MAX_K_TO_TEST` at the top of the script to
  experiment (the slide deck explicitly encourages this).
- Swap `KNeighborsClassifier` for another `sklearn` classifier (e.g. `LogisticRegression`,
  `DecisionTreeClassifier`) to compare algorithms, as suggested in the conclusion slide.

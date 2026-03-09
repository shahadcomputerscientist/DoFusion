# DocFusion: Intelligent Receipt Processing

This project was developed for the **2026 ML Rihal CodeStacker Challenge**.

The goal is to build an intelligent document processing pipeline that can:

- Extract structured information from receipts
- Detect forged or suspicious documents
- Provide a simple web interface for users

---

# Project Pipeline

The system processes scanned receipts using the following pipeline:

Receipt Image ->  OCR (PaddleOCR) -> Text & Feature Extraction -> Machine Learning Model (Random Forest) -> Forgery Prediction  

---

# Extracted Fields

The system extracts the following structured data:

- **Vendor** – merchant or company name
- **Date** – transaction date
- **Total** – final receipt amount

Additional features used for anomaly detection:

- number of text lines
- text length
- number of price values
- average price
- maximum price
- price consistency

---

# Forgery Detection

The system detects suspicious receipts using a **Random Forest classifier** trained on the **Find-It-Again dataset**.

Possible indicators of fraud include:

- inconsistent totals
- unusual price distributions
- abnormal receipt structure

---

# Datasets Used

This project integrates multiple datasets:

### SROIE Dataset
Used for receipt structure and OCR evaluation.

### Find-It-Again Dataset
Contains forged and genuine receipts used for anomaly detection training.

### CORD Dataset
Used to test robustness against different layouts and noise.

---

# Running the Pipeline

Train the model and generate predictions:

```bash
python test_sol.py

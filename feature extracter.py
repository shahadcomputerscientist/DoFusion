import pandas as pd
from paddleocr import PaddleOCR
import re
import os

ocr = PaddleOCR(lang='en')

folder = "data/findit2/train"
features = []

for filename in os.listdir(folder):

    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(folder, filename)

    result = ocr.ocr(img_path)

    if result is None or len(result) == 0:
        continue

    # Extract text lines
    text_lines = [line[1][0] for line in result[0]]

    if len(text_lines) == 0:
        continue

    invalid_vendor_words = ["TAX", "INVOICE", "RECEIPT"]

    vendor = text_lines[0] if len(text_lines) > 0 else None

    if vendor and any(w in vendor.upper() for w in invalid_vendor_words):
        vendor = text_lines[1] if len(text_lines) > 1 else vendor

    if vendor:
        vendor = vendor.strip().upper()

    # ------------------------
    # Date extraction
    # ------------------------

    date = None

    date_patterns = [
        r"\d{2}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}\.\d{2}\.\d{2}"
    ]

    prices = []

    for line in text_lines:

        # Detect dates
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                date = match.group()

        # Detect prices like 23.40 or 75,000
        m = re.search(r"\d+[.,]\d{2,3}", line)

        if m:
            value = float(m.group().replace(",", ""))
            prices.append(value)

    if prices:
        total = max(prices)
    else:
        total = None

    numbers_of_lines = len(text_lines)

    text_length = sum(len(line) for line in text_lines)

    total_per_line = total / numbers_of_lines if total and numbers_of_lines > 0 else None

    features.append({
        "file": filename,
        "vendor": vendor,
        "date": date,
        "total": total,
        "numbers_of_lines": numbers_of_lines,
        "text_length": text_length,
        "total_per_line": total_per_line
    })

df = pd.DataFrame(features)
df.to_csv("findit2_features.csv", index=False)

print("Feature extraction completed.")
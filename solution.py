import json
from paddleocr import PaddleOCR
import re
import os

class DocFusionSolution:
    def __init__(self):
        self.ocr = PaddleOCR(lang="en",use_angle_cls=False)

    def extract_features(self, image_path):
        result = self.ocr.ocr(image_path)

        if result is None or len(result) == 0:
            return None, None, None, 0,0

        # Extract text lines
        text_lines = [line[1][0] for line in result[0]]

        if len(text_lines) == 0:
            return None, None, None, 0,0

        invalid_vendor_words = ["TAX", "INVOICE", "RECEIPT"]

        vendor = text_lines[0] if len(text_lines) > 0 else None

        if vendor and any(w in vendor.upper() for w in invalid_vendor_words):
            vendor = text_lines[1] if len(text_lines) > 1 else vendor

        if vendor:
            vendor = vendor.strip().upper()

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
        num_prices = len(prices)
        avg_price = sum(prices) / len(prices) if prices else 0
        max_price = max(prices) if prices else 0
        sum_prices = sum(prices) if prices else 0
        if total:
            total_diff = abs(sum_prices - total)
        else:
            total_diff = 0

        return vendor, date, total, numbers_of_lines, text_length, num_prices, avg_price, max_price, total_diff

    def train(self, train_dir: str, work_dir: str) -> str:
        """
        Train a model on data in train_dir.

        Args:
            train_dir: Path to directory containing train.jsonl and images/
            work_dir:  Scratch directory for writing model artifacts

        Returns:
            Path to the saved model directory (typically inside work_dir)
        """
        from sklearn.ensemble import RandomForestClassifier  # very efficient with tabular data
        import joblib
        train_json = os.path.join(train_dir, "train.jsonl")
        images_dir = os.path.join(train_dir, "images/training")

        X=[]
        y=[]
        with open(train_json) as f:
            data =[]
            for line in f:
                data.append(json.loads(line))

        for item in data:
            img_id = item["id"]
            img_label = item.get("is_forged", 0)
            img_path = os.path.join(images_dir, img_id + ".png")

            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, img_id + ".jpg")

            vendor, date, total, numbers_of_lines, text_length, num_prices, avg_price, max_price, total_diff= self.extract_features(img_path)
            total = total if total else 0
            X.append([total, numbers_of_lines, text_length,num_prices, avg_price, max_price, total_diff])
            y.append(img_label)
        model = RandomForestClassifier(n_estimators=100)  # create model
        model.fit(X, y)  # train it on the dataset

        os.makedirs(work_dir, exist_ok=True)
        model_path = os.path.join(work_dir, "model.pkl")
        joblib.dump(model, model_path)

        return work_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """
        Run inference and write predictions to out_path.

        Args:
            model_dir: Path returned by train()
            data_dir:  Path to directory containing test.jsonl and images/
            out_path:  Path where predictions JSONL should be written
        """
        import joblib
        # load the model
        model_path = os.path.join(model_dir, "model.pkl")
        model = joblib.load(model_path)
        test_json = os.path.join(data_dir, "test.jsonl")
        images_dir = os.path.join(data_dir, "images/test")
        with open(test_json) as f:
            data =[]
            for line in f:
                data.append(json.loads(line))
        predictions = []
        for item in data:
            img_id = item["id"]
            img_path = os.path.join(images_dir, img_id + ".png")

            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, img_id + ".jpg")

            vendor, date, total, numbers_of_lines, text_length, num_prices, avg_price, max_price, total_diff = self.extract_features(img_path)
            total = total if total else 0
            features = [[total, numbers_of_lines, text_length,  num_prices, avg_price, max_price, total_diff]]
            is_forged = int(model.predict(features)[0])
            predictions.append({
                "id": img_id,
                "vendor": vendor,
                "date": date,
                "total": str(total) if total else None,
                "is_forged": is_forged
            })
        with open(out_path, "w") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction) + "\n")
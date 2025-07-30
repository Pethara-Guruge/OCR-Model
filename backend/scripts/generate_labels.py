import pandas as pd
from pathlib import Path
import os
import cv2

try:
    import pytesseract
except ImportError:
    pytesseract = None

def extract_text(img_path):
    """Basic OCR function using pytesseract"""
    if pytesseract is None:
        return ""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        print(f"OCR failed for {img_path}: {str(e)}")
        return ""

def generate_labels(use_ocr=True):
    raw_data_path = Path("backend/data/raw/set/data")
    output_path = Path("backend/data/splits/train/labels.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for folder in raw_data_path.glob("[0-9][0-9][0-9]"):
        if not folder.is_dir():
            continue
        for img_file in folder.glob("*.png"):
            record = {
                "image_path": str(Path("..") / "raw" / "set" / "data" / folder.name / img_file.name),
                "writer_id": folder.name
            }

            full_img_path = folder / img_file.name  # Ensure proper path for OCR

            if use_ocr:
                print(f"Running OCR on {full_img_path}")
                extracted_text = extract_text(full_img_path)
                record.update({
                    "text_auto": extracted_text,
                    "text_manual": ""
                })
            else:
                print(f"Manual entry required for {full_img_path}")
                record["text"] = ""

            records.append(record)

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated labels file at: {output_path}")
    print(f"Total samples: {len(df)}")

    if use_ocr:
        print("Auto-generated text may need manual correction!")
    else:
        if "text" in df.columns and (df["text"].isnull().any() or df["text"].eq('').any()):
            print("Please manually fill in the 'text' column.")
        else:
            print("No manual corrections needed.")

if __name__ == "__main__":
    generate_labels(use_ocr=True)
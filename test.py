import cv2
import pandas as pd
import numpy as np
import re
from datetime import datetime
import json
import os
import requests
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io


class ImprovedGoodReceiptOCR:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """
        Improved OCR dengan debugging dan multiple extraction methods
        """
        self.ollama_host = ollama_host
        self.model_name = "llama3.2:3b"

        # Initialize EasyOCR
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en', 'id'], gpu=False)
            self.use_easyocr = True
            print("✓ EasyOCR initialized successfully")
        except ImportError:
            print("❌ EasyOCR not installed. Run: pip install easyocr")
            self.use_easyocr = False

        # Improved patterns untuk dokumen Good Receipt Indonesia
        self.patterns = {
            'vendor_patterns': [
                r'PT\s+([A-Za-z\s]+?)(?:\s+Alamat|Alamat|\s+Kota|\s+Provinsi|$)',  # PT + nama saja
                r'CV\s+([A-Za-z\s]+?)(?:\s+Alamat|Alamat|\s+Kota|\s+Provinsi|$)',  # CV + nama saja
                r'(?:Nama Vendor|Vendor)[\s:]+PT\s+([A-Za-z\s]+?)(?:\s+Alamat|$)',
                r'(?:RECEIVED FROM|dari)[\s:]+PT\s+([A-Za-z\s]+?)(?:\s+Alamat|$)',
            ],
            'date_patterns': [
                r'DATE[\s:]+(\d{1,2}[\s\-\/]\w+[\s\-\/]\d{4})',
                r'(\d{1,2}\s+\w+\s+\d{4})',
                r'(\d{1,2}[\-\/]\d{1,2}[\-\/]\d{4})',
                r'tanggal[\s:]+(\d{1,2}[\s\-\/]\w+[\s\-\/]\d{4})',
            ],
            'po_patterns': [
                r'(?:OUR P\.O\.|P\.O\.|PO)[\s#:]*([A-Z0-9\-]+)',
                r'PO[\s\-#:]*([A-Z0-9\-]+\d+)',
                r'Purchase Order[\s:]+([A-Z0-9\-]+)',
                r'(?:id|ID)\s*po[\s:]*([A-Z0-9\-]+)',  # untuk "id po"
            ],
            'bill_patterns': [
                r'(?:Bill of Lading|Freight Bill)[\s#:]*([A-Z0-9]+)',
                r'#[\s]*([A-Z]{2,3}\d{4,6})',
                r'No[\s\.]*([A-Z0-9\-]+)',
            ]
        }

    def preprocess_image_advanced(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced preprocessing untuk berbagai kondisi dokumen
        """
        try:
            # Read original image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocessing version 1: Standard
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Adaptive thresholding
            thresh1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Preprocessing version 2: Enhanced for handwriting
            # Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # OTSU thresholding
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

            # Resize untuk OCR optimal
            height, width = cleaned.shape
            if height < 1000:
                scale_factor = 1000 / height
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                thresh1 = cv2.resize(thresh1, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            return thresh1, cleaned

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None

    def extract_text_multiple_methods(self, image_path: str) -> List[str]:
        """
        Extract text menggunakan multiple methods untuk comparison
        """
        all_texts = []

        # Method 1: EasyOCR on original
        if self.use_easyocr:
            try:
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                print("🔍 Method 1: EasyOCR on original...")
                results = self.ocr_reader.readtext(img_rgb, detail=0, paragraph=True)
                text1 = '\n'.join(results)
                all_texts.append(("EasyOCR Original", text1))

            except Exception as e:
                print(f"EasyOCR Method 1 failed: {e}")

        # Method 2: EasyOCR on preprocessed images
        thresh1, thresh2 = self.preprocess_image_advanced(image_path)
        if thresh1 is not None and self.use_easyocr:
            try:
                print("🔍 Method 2: EasyOCR on preprocessed...")
                results = self.ocr_reader.readtext(thresh1, detail=0, paragraph=True)
                text2 = '\n'.join(results)
                all_texts.append(("EasyOCR Preprocessed 1", text2))

                results = self.ocr_reader.readtext(thresh2, detail=0, paragraph=True)
                text3 = '\n'.join(results)
                all_texts.append(("EasyOCR Preprocessed 2", text3))

            except Exception as e:
                print(f"EasyOCR Method 2 failed: {e}")

        # Method 3: Try with different EasyOCR settings
        if self.use_easyocr:
            try:
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                print("🔍 Method 3: EasyOCR with different settings...")
                results = self.ocr_reader.readtext(
                    img_rgb,
                    detail=0,
                    paragraph=False,
                    width_ths=0.7,
                    height_ths=0.7
                )
                text4 = '\n'.join(results)
                all_texts.append(("EasyOCR Settings Tuned", text4))

            except Exception as e:
                print(f"EasyOCR Method 3 failed: {e}")

        return all_texts

    def debug_extraction(self, all_texts: List[Tuple[str, str]]):
        """
        Debug extracted texts
        """
        print("\n=== DEBUGGING EXTRACTED TEXTS ===")
        for method, text in all_texts:
            print(f"\n--- {method} ---")
            print(f"Length: {len(text)} characters")
            print(f"Preview: {text[:300]}...")
            print("-" * 40)

    def extract_field_from_multiple_texts(self, all_texts: List[Tuple[str, str]], field_name: str) -> str:
        """
        Extract specific field dari multiple text results
        """
        patterns = self.patterns.get(f'{field_name}_patterns', [])

        for method, text in all_texts:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1).strip()
                    print(f"✓ Found {field_name} with {method}: {result}")
                    return result

        print(f"❌ {field_name} not found in any method")
        return ""

    def extract_items_from_multiple_texts(self, all_texts: List[Tuple[str, str]]) -> List[Dict]:
        """
        Extract items table dari multiple texts
        """
        items = []

        for method, text in all_texts:
            print(f"\n🔍 Looking for items in {method}...")

            # Split ke lines
            lines = text.split('\n')

            # Look for table headers
            table_start = -1
            for i, line in enumerate(lines):
                if re.search(r'(?:QUANTITY|QTY).*(?:DESCRIPTION|DESC)', line, re.IGNORECASE):
                    table_start = i + 1
                    print(f"✓ Found table header at line {i}")
                    break

            if table_start == -1:
                # Alternative: look for numeric + text patterns
                for i, line in enumerate(lines):
                    # Pattern: angka + teks + kondisi + berat + unit
                    if re.search(r'^\s*\d+\s+[A-Za-z]', line):
                        items_found = self.extract_items_from_lines(lines[i:])
                        if items_found:
                            print(f"✓ Found {len(items_found)} items with pattern matching")
                            return items_found
                continue

            # Extract dari table
            items_found = self.extract_items_from_lines(lines[table_start:])
            if items_found:
                print(f"✓ Found {len(items_found)} items in {method}")
                return items_found

        print("❌ No items found in any method")
        return items

    def extract_items_from_lines(self, lines: List[str]) -> List[Dict]:
        """
        Extract items dengan focus pada quantity, description, dan weight saja
        """
        items = []

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Pattern 1: qty + description + weight/details
            # Contoh: "5 Laptop Dell Inspiron Baik 10 Kg Masuk"
            pattern1 = r'(\d+)\s+(.+?)\s+(?:Baik|Good|OK)?\s*(\d+)\s*(?:Kg|kg|KG)'
            match = re.search(pattern1, line)

            if match:
                items.append({
                    'quantity': match.group(1),
                    'description': match.group(2).strip(),
                    'weight': match.group(3) + ' Kg'
                })
                continue

            # Pattern 2: qty + description (tanpa weight yang jelas)
            # Contoh: "3 Printer HP LaserJet Baik"
        pattern2 = r'(\d+)\s+(.+?)(?:\s+(?:Baik|Good|OK|Bad))?

    def process_document(self, image_path: str) -> Dict:
        """
        Process dokumen dengan improved method
        """
        print(f"Processing document: {image_path}")

        # Extract text menggunakan multiple methods
        all_texts = self.extract_text_multiple_methods(image_path)

        if not all_texts:
            return {"error": "Failed to extract any text"}

        # Debug extracted texts
        self.debug_extraction(all_texts)

        # Extract individual fields
        result = {
            'vendor_name': self.extract_field_from_multiple_texts(all_texts, 'vendor'),
            'date': self.extract_field_from_multiple_texts(all_texts, 'date'),
            'po_number': self.extract_field_from_multiple_texts(all_texts, 'po'),
            'bill_number': self.extract_field_from_multiple_texts(all_texts, 'bill'),
            'items': self.extract_items_from_multiple_texts(all_texts),
            'all_extracted_texts': all_texts  # For debugging
        }
        return result

    def save_to_excel(self, extracted_data: Dict, output_path: str = "good_receipt_clean.xlsx"):
        """
        Save semua data dalam 1 sheet saja dengan format yang clean
        """
        try:
            # Prepare data untuk single sheet
            output_data = []

            # Row 1: Header info
            output_data.append({
                'Field': 'Vendor Name',
                'Value': extracted_data.get('vendor_name', ''),
                'Quantity': '',
                'Description': '',
                'Weight': ''
            })

            output_data.append({
                'Field': 'Date',
                'Value': extracted_data.get('date', ''),
                'Quantity': '',
                'Description': '',
                'Weight': ''
            })

            output_data.append({
                'Field': 'PO Number',
                'Value': extracted_data.get('po_number', ''),
                'Quantity': '',
                'Description': '',
                'Weight': ''
            })

            output_data.append({
                'Field': 'Bill Number',
                'Value': extracted_data.get('bill_number', ''),
                'Quantity': '',
                'Description': '',
                'Weight': ''
            })

            # Empty row
            output_data.append({
                'Field': '',
                'Value': '',
                'Quantity': '',
                'Description': '',
                'Weight': ''
            })

            # Items header
            output_data.append({
                'Field': 'ITEMS',
                'Value': 'LIST',
                'Quantity': 'Quantity',
                'Description': 'Description',
                'Weight': 'Weight'
            })

            # Items data
            items = extracted_data.get('items', [])
            if items:
                for i, item in enumerate(items, 1):
                    output_data.append({
                        'Field': f'Item {i}',
                        'Value': '',
                        'Quantity': item.get('quantity', ''),
                        'Description': item.get('description', ''),
                        'Weight': item.get('weight', '')
                    })
            else:
                output_data.append({
                    'Field': 'No items',
                    'Value': 'detected',
                    'Quantity': '',
                    'Description': '',
                    'Weight': ''
                })

            # Create DataFrame
            df = pd.DataFrame(output_data)

            # Save to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Good_Receipt_Data', index=False)

                # Format the Excel
                workbook = writer.book
                worksheet = writer.sheets['Good_Receipt_Data']

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            print(f"✓ Data saved to {output_path} in single sheet format")
            return True

        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            return False


def main():
    """
    Main function dengan improved debugging
    """
    print("=== Improved Good Receipt OCR ===")

    # Check EasyOCR
    try:
        import easyocr
        print("✓ EasyOCR available")
    except ImportError:
        print("❌ EasyOCR not installed")
        print("💡 Install with: pip install easyocr")
        return

    # Initialize
    ocr_detector = ImprovedGoodReceiptOCR()

    # Process
    image_path = "dummy.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    result = ocr_detector.process_document(image_path)

    # Results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("\n=== FINAL EXTRACTION RESULTS ===")
        print(f"📋 Vendor Name: '{result.get('vendor_name', 'Not found')}'")
        print(f"📅 Date: '{result.get('date', 'Not found')}'")
        print(f"🔢 PO Number: '{result.get('po_number', 'Not found')}'")
        print(f"📄 Bill Number: '{result.get('bill_number', 'Not found')}'")
        print(f"📦 Items Found: {len(result.get('items', []))}")

        for i, item in enumerate(result.get('items', []), 1):
            print(f"\n  Item {i}:")
            for key, value in item.items():
                print(f"    {key}: '{value}'")

        # Save
        ocr_detector.save_to_excel(result, "good_receipt_clean.xlsx")
        print("\n✅ Data saved in single clean sheet format!")


if __name__ == "__main__":
    main()

    match = re.search(pattern2, line)

    if match and len(match.group(2).strip()) > 5:
        desc = match.group(2).strip()
        # Filter out non-item lines (tanggal, total, dll)
        if not re.search(r'(?:total|subtotal|tax|amount|date|vendor|alamat|kota|provinsi|received|charges|delivery)',
                         desc, re.IGNORECASE):
            # Extract weight dari description jika ada
            weight_match = re.search(r'(\d+)\s*(?:kg|Kg|KG)', desc)
            weight = weight_match.group(0) if weight_match else ''

            # Clean description dari weight info
            clean_desc = re.sub(r'\s*\d+\s*(?:kg|Kg|KG)\s*', ' ', desc).strip()

            items.append({
                'quantity': match.group(1),
                'description': clean_desc,
                'weight': weight
            })

return items


def process_document(self, image_path: str) -> Dict:
    """
    Process dokumen dengan improved method
    """
    print(f"Processing document: {image_path}")

    # Extract text menggunakan multiple methods
    all_texts = self.extract_text_multiple_methods(image_path)

    if not all_texts:
        return {"error": "Failed to extract any text"}

    # Debug extracted texts
    self.debug_extraction(all_texts)

    # Extract individual fields
    result = {
        'vendor_name': self.extract_field_from_multiple_texts(all_texts, 'vendor'),
        'date': self.extract_field_from_multiple_texts(all_texts, 'date'),
        'po_number': self.extract_field_from_multiple_texts(all_texts, 'po'),
        'bill_number': self.extract_field_from_multiple_texts(all_texts, 'bill'),
        'items': self.extract_items_from_multiple_texts(all_texts),
        'all_extracted_texts': all_texts  # For debugging
    }

    return result


def save_to_excel(self, extracted_data: Dict, output_path: str = "good_receipt_improved.xlsx"):
    """
    Save extracted data ke Excel dengan debugging info
    """
    try:
        # Main info
        main_info = pd.DataFrame([{
            'Vendor Name': extracted_data.get('vendor_name', ''),
            'Date': extracted_data.get('date', ''),
            'PO Number': extracted_data.get('po_number', ''),
            'Bill Number': extracted_data.get('bill_number', ''),
            'Total Items': len(extracted_data.get('items', [])),
            'Processing Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])

        # Items
        items_data = extracted_data.get('items', [])
        if items_data:
            items_df = pd.DataFrame(items_data)
        else:
            items_df = pd.DataFrame(columns=['quantity', 'description', 'uom', 'condition', 'weight', 'status'])

        # Debug info
        debug_data = []
        for method, text in extracted_data.get('all_extracted_texts', []):
            debug_data.append({
                'Method': method,
                'Text_Length': len(text),
                'Text_Preview': text[:500]
            })
        debug_df = pd.DataFrame(debug_data)

        # Save to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            main_info.to_excel(writer, sheet_name='Document_Info', index=False)
            items_df.to_excel(writer, sheet_name='Items_List', index=False)
            debug_df.to_excel(writer, sheet_name='Debug_Info', index=False)

        print(f"✓ Data saved to {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")
        return False


def main():
    """
    Main function dengan improved debugging
    """
    print("=== Improved Good Receipt OCR ===")

    # Check EasyOCR
    try:
        import easyocr
        print("✓ EasyOCR available")
    except ImportError:
        print("❌ EasyOCR not installed")
        print("💡 Install with: pip install easyocr")
        return

    # Initialize
    ocr_detector = ImprovedGoodReceiptOCR()

    # Process
    image_path = "dummy.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    result = ocr_detector.process_document(image_path)

    # Results
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("\n=== FINAL EXTRACTION RESULTS ===")
        print(f"📋 Vendor Name: '{result.get('vendor_name', 'Not found')}'")
        print(f"📅 Date: '{result.get('date', 'Not found')}'")
        print(f"🔢 PO Number: '{result.get('po_number', 'Not found')}'")
        print(f"📄 Bill Number: '{result.get('bill_number', 'Not found')}'")
        print(f"📦 Items Found: {len(result.get('items', []))}")

        for i, item in enumerate(result.get('items', []), 1):
            print(f"\n  Item {i}:")
            for key, value in item.items():
                print(f"    {key}: '{value}'")

        # Save
        ocr_detector.save_to_excel(result, "good_receipt_improved.xlsx")
        print("\n✅ Check the Excel file for detailed debugging info!")


if __name__ == "__main__":
    main()
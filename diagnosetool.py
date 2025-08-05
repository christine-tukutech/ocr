"""
OCR Processor - Extract to Structured Excel with 8 Columns
Hybrid approach: Tesseract + Pattern Recognition
"""

import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from typing import Dict, List, Tuple
from pathlib import Path


class StructuredOCRProcessor:
    def __init__(self):
        self.output_dir = "result"
        os.makedirs(self.output_dir, exist_ok=True)

        # Pattern definitions
        self.vendor_patterns = [
            r'PT[\.\s]+([A-Z][A-Za-z\s&]+?)(?:\n|$|,)',
            r'CV[\.\s]+([A-Z][A-Za-z\s&]+?)(?:\n|$|,)',
            r'UD[\.\s]+([A-Z][A-Za-z\s&]+?)(?:\n|$|,)',
            r'^([A-Z][A-Za-z\s&]+?)\s*(?:warehouse|gudang)',
        ]

        self.recipient_patterns = [
            r'Toko\s+Kopi\s+([^\n]+)',
            r'(?:ship\s*to|kirim\s*ke|tujuan)[\s:]+([^\n]+)',
            r'(?:customer|pelanggan)[\s:]+([^\n]+)',
            r'Toko\s+([^\n]+)',
        ]

        self.date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
            r'(?:date|tanggal|tgl)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        ]

        # Units of Measure
        self.uom_list = [
            'PACK', 'PCS', 'PC', 'BOX', 'CTN', 'CARTON', 'KG', 'GRAM', 'GR',
            'LT', 'LITER', 'ML', 'UNIT', 'UN', 'BTL', 'BOTTLE', 'ROLL', 'ROL',
            'DUS', 'LUSIN', 'DOZEN', 'SET', 'LEMBAR', 'SHEET', 'GALON', 'GALLON'
        ]

    def preprocess_image(self, image_path):
        """Enhanced preprocessing for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if too small
        height, width = gray.shape
        if width < 1500:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check if inverted
        white_pixels = cv2.countNonZero(thresh)
        if white_pixels < thresh.size * 0.5:
            thresh = cv2.bitwise_not(thresh)

        return thresh

    def extract_vendor(self, text):
        """Extract vendor name (PT/CV/UD)"""
        for pattern in self.vendor_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                vendor = match.group(0).strip()
                # Clean up
                vendor = re.sub(r'\s+', ' ', vendor)
                return vendor
        return ""

    def extract_recipient(self, text):
        """Extract recipient/delivery destination"""
        for pattern in self.recipient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                recipient = match.group(1) if match.lastindex else match.group(0)
                recipient = recipient.strip()
                # Clean up
                recipient = re.sub(r'\s+', ' ', recipient)
                return recipient
        return ""

    def extract_date(self, text):
        """Extract date from text"""
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        return ""

    def extract_po_number(self, text):
        """Extract PO number"""
        po_patterns = [
            r'(?:PO|P\.O\.|Purchase\s*Order)[\s:#]*([A-Z0-9\-/]+)',
            r'No[\.\s]+PO[\s:]+([A-Z0-9\-/]+)',
            r'PO\s*Number[\s:]+([A-Z0-9\-/]+)',
        ]

        for pattern in po_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def extract_reference_number(self, text):
        """Extract reference number"""
        ref_patterns = [
            r'Reference\s*Number[\s:]+(\d+)',
            r'Ref\s*(?:No|Number)[\s:]+([A-Z0-9\-/]+)',
            r'(?:SPB|RSO)\s*Number[\s:]+([A-Z0-9\-/]+)',
            r'Reference[\s:]+(\d+)',
        ]

        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def extract_items_advanced(self, text):
        """Advanced item extraction with multiple strategies"""
        items = []
        lines = text.split('\n')

        # Strategy 1: Look for item patterns with codes
        item_patterns = [
            # Pattern: CODE: Description @price qty unit
            r'([A-Z]{2}\d{4,})[:\s]+(.+?)\s+@?\s*\$?\s*\d*\.?\d*\s+#?\s*(\d+(?:\.\d+)?)\s+(\w+)',
            # Pattern: CODE - Description qty unit
            r'([A-Z]{2}\d{4,})\s*[-:]\s*(.+?)\s+(\d+(?:\.\d+)?)\s+(\w+)',
            # Pattern: Number Item_Name Qty Unit
            r'^(\d{1,3})\s+(.+?)\s+(\d+(?:\.\d+)?)\s+(PACK|PCS|KG|LT|ML|UNIT|BTL|ROLL|BOX|CTN)',
            # Pattern: Item_Name Qty Unit (no number)
            r'^([A-Za-z].+?)\s+(\d+(?:\.\d+)?)\s+(PACK|PCS|KG|LT|ML|UNIT|BTL|ROLL|BOX|CTN)',
        ]

        # Process each line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip headers
            if any(header in line.upper() for header in ['ITEM', 'DESCRIPTION', 'NO.', 'QUANTITY', 'UOM']):
                continue

            # Try each pattern
            for pattern in item_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()

                    if len(groups) == 4:  # Has item code/number
                        item_data = {
                            'no_item': groups[0],
                            'item': groups[1].strip(),
                            'qty': groups[2],
                            'uom': groups[3].upper()
                        }
                    elif len(groups) == 3:  # No item number
                        item_data = {
                            'no_item': str(len(items) + 1),
                            'item': groups[0].strip(),
                            'qty': groups[1],
                            'uom': groups[2].upper()
                        }
                    else:
                        continue

                    # Validate and clean
                    if len(item_data['item']) > 3 and float(item_data['qty']) > 0:
                        items.append(item_data)
                        break

        # Strategy 2: Table detection with quantities
        if len(items) < 3:  # If few items found, try alternative approach
            # Look for lines with quantities and units
            qty_pattern = r'(\d+(?:[.,]\d+)?)\s*(' + '|'.join(self.uom_list) + r')'

            for i, line in enumerate(lines):
                matches = re.findall(qty_pattern, line, re.IGNORECASE)

                for qty, unit in matches:
                    # Try to find item name before the quantity
                    before_match = line[:line.lower().find(qty)]

                    # Extract potential item name
                    item_match = re.search(r'([A-Za-z][A-Za-z0-9\s\-\.]+?)(?:\s+\d|$)', before_match)
                    if item_match:
                        item_name = item_match.group(1).strip()

                        # Look for item number at start of line
                        no_match = re.match(r'^(\d{1,3}|\w{2}\d{4,})', line)
                        item_no = no_match.group(1) if no_match else str(len(items) + 1)

                        if len(item_name) > 3:
                            items.append({
                                'no_item': item_no,
                                'item': item_name,
                                'qty': qty.replace(',', '.'),
                                'uom': unit.upper()
                            })

        # Strategy 3: Handle multi-column quantity formats (ordered/prepared/received)
        multi_qty_pattern = r'(.+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)'

        for line in lines:
            if 'ordered' in line.lower() or 'prepared' in line.lower() or 'received' in line.lower():
                # Found header, process next lines
                continue

            match = re.search(multi_qty_pattern, line)
            if match and not any(item['item'] == match.group(1).strip() for item in items):
                # Use the last quantity (usually received)
                qty = match.group(4) if match.group(4) != '0' else match.group(3)

                # Try to find unit
                unit_match = re.search(r'(' + '|'.join(self.uom_list) + r')', line, re.IGNORECASE)
                unit = unit_match.group(1).upper() if unit_match else 'UNIT'

                items.append({
                    'no_item': str(len(items) + 1),
                    'item': match.group(1).strip(),
                    'qty': qty,
                    'uom': unit
                })

        return items

    def process_image(self, image_path):
        """Process single image and extract all required fields"""
        filename = os.path.basename(image_path)
        print(f"\n📄 Processing: {filename}")

        try:
            # Preprocess
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                print(f"  ❌ Failed to read image")
                return None

            # OCR with Indonesian + English
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_img, lang='ind+eng', config=custom_config)

            # Extract all fields
            data = {
                'filename': filename,
                'tanggal': self.extract_date(text),
                'nama_vendor': self.extract_vendor(text),
                'tujuan_pengiriman': self.extract_recipient(text),
                'no_po': self.extract_po_number(text),
                'reference_number': self.extract_reference_number(text),
                'items': self.extract_items_advanced(text)
            }

            # Display results
            print(f"  ✅ Tanggal: {data['tanggal'] or 'Not found'}")
            print(f"  ✅ Vendor: {data['nama_vendor'] or 'Not found'}")
            print(f"  ✅ Tujuan: {data['tujuan_pengiriman'] or 'Not found'}")
            print(f"  ✅ PO: {data['no_po'] or 'Not found'}")
            print(f"  ✅ Ref: {data['reference_number'] or 'Not found'}")
            print(f"  ✅ Items: {len(data['items'])} found")

            # Save raw text for debugging
            txt_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Extracted from: {filename}\n")
                f.write(f"Date: {datetime.now()}\n")
                f.write("=" * 50 + "\n")
                f.write(text)

            return data

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return None

    def save_to_excel(self, all_data):
        """Save all extracted data to Excel with proper structure"""
        # Prepare rows for Excel
        excel_rows = []

        for file_data in all_data:
            if not file_data:
                continue

            # Base information
            base_info = {
                'Tanggal': file_data['tanggal'],
                'Nama_Vendor': file_data['nama_vendor'],
                'Tujuan_Pengiriman': file_data['tujuan_pengiriman'],
                'No_PO': file_data['no_po'],
                'Reference_Number': file_data['reference_number'],
                'Source_File': file_data['filename']
            }

            # If no items, add one row with base info
            if not file_data['items']:
                row = base_info.copy()
                row.update({
                    'No_Item': '',
                    'Item': '',
                    'UOM': '',
                    'QTY': ''
                })
                excel_rows.append(row)
            else:
                # Add row for each item
                for item in file_data['items']:
                    row = base_info.copy()
                    row.update({
                        'No_Item': item['no_item'],
                        'Item': item['item'],
                        'UOM': item['uom'],
                        'QTY': item['qty']
                    })
                    excel_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(excel_rows)

        # Reorder columns as specified
        column_order = [
            'Tanggal', 'Nama_Vendor', 'Tujuan_Pengiriman', 'No_PO',
            'Reference_Number', 'No_Item', 'Item', 'UOM', 'QTY', 'Source_File'
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = ''

        df = df[column_order]

        # Save to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.output_dir, f"OCR_Results_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Extracted_Data', index=False)

            # Summary sheet
            summary_data = {
                'Metric': ['Total Files', 'Total Items', 'Files with Vendor', 'Files with Date', 'Files with Items'],
                'Count': [
                    len(all_data),
                    len(df),
                    df['Nama_Vendor'].notna().sum(),
                    df['Tanggal'].notna().sum(),
                    df['Item'].notna().sum()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
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

        print(f"\n💾 Excel saved: {excel_path}")
        return excel_path

    def process_folder(self, folder_path="gambar"):
        """Process all images in folder"""
        print("\n🔍 STRUCTURED OCR PROCESSOR")
        print("📊 Extracting 8 required columns")
        print("=" * 60)

        if not os.path.exists(folder_path):
            print(f"❌ Folder '{folder_path}' not found!")
            return

        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, file))

        if not image_files:
            print(f"❌ No images found in folder")
            return

        image_files.sort()
        print(f"📁 Found {len(image_files)} images to process")
        print("=" * 60)

        # Process all images
        all_results = []

        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}]", end="")
            result = self.process_image(image_path)
            if result:
                all_results.append(result)

        # Save to Excel
        if all_results:
            excel_path = self.save_to_excel(all_results)

            print("\n" + "=" * 60)
            print(f"✅ PROCESSING COMPLETE")
            print(f"📊 Processed: {len(all_results)}/{len(image_files)} files")
            print(f"📁 Results folder: '{self.output_dir}'")
            print(f"📑 Excel file: {os.path.basename(excel_path)}")
            print("\n📋 Excel contains 8 columns as requested:")
            print("   1. Tanggal")
            print("   2. Nama_Vendor (PT/CV/UD)")
            print("   3. Tujuan_Pengiriman (Toko Kopi, etc)")
            print("   4. No_PO")
            print("   5. Reference_Number")
            print("   6. No_Item")
            print("   7. Item")
            print("   8. UOM")
            print("   9. QTY")
            print("   10. Source_File (for reference)")
            print("=" * 60)
        else:
            print("\n❌ No data extracted")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Check dependencies
    try:
        import pytesseract
        import cv2
        import pandas
    except ImportError:
        print("📦 Installing dependencies...")
        os.system("pip install pytesseract opencv-python pandas openpyxl")
        print("✅ Dependencies installed. Please run the script again.")
        exit()

    # Run processor
    processor = StructuredOCRProcessor()
    processor.process_folder("gambar")

    print("\n🎉 DONE! Check 'result' folder for:")
    print("   - Individual .txt files (raw OCR)")
    print("   - Combined Excel file with all data")
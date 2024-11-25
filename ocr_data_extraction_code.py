# IMPORT ALL LIBRARIES
import os
import string
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
import pytesseract
import pdf2image
import re
import multiprocessing
from PyPDF2 import PdfReader
import pandas as pd
import concurrent.futures
import time
from datetime import datetime
import pyodbc
import concurrent.futures
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import easyocr
from rapidfuzz import process
import cv2
from paddleocr import PaddleOCR
import logging
from rapidfuzz import fuzz, process
# Note: Change all paths, database name, table name, and server IP address as needed.

# Setting up Tesseract and Poppler paths
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
base_dir = r"D:\PDFServicesSDK-PythonSamples\adobe-dc-pdf-services-sdk-python\src\ocrpdf"
os.chdir(base_dir)
poppler_relative_path = r"D:\PYTHON DEVLOPMENT\OCR\OCR_voters-pdf_data_extraction\poppler\Library\bin"
poppler_path = os.path.join(base_dir, poppler_relative_path)

# Paths for PDF input and output directories
pdf_folder = r"D:\PDFServicesSDK-PythonSamples\output_pdf"
output_dir = r"D:\PDFServicesSDK-PythonSamples\output_excel"

# Save directory for debugging purposes
media_folder = r"D:\PYTHON DEVELOPMENT\OCR\OCR_voters-pdf_data_extraction\media"
os.makedirs(media_folder, exist_ok=True)
# Database connection string and table name
connection_string = 'DRIVER={SQL Server};SERVER=WORKSTATION23;DATABASE=MH_AC_134_Aug2024_ocr;UID=sa;PWD=manager'
table_name = 'Ac_134_VotersInfo'

# List of known correct prefixes


# def correct_common_ocr_errors(text, correct_prefixes):
#     # Common patterns that OCR might misread
#     likely_errors = {
#         '0': 'D', 'D': '0',  # Zero and letter D
#         '5': 'S', 'S': '5',  # Number five and letter S
#         '1': 'I', 'I': '1',  # Number one and letter I
#         '8': 'B', 'B': '8',  # Number eight and letter B
#         '6': 'G', 'G': '6',  # Number six and letter G 
#         '2': 'Z', 'Z': '2'   # Number two and letter Z
#     }

#     # A dictionary to hold corrections
#     corrections = {}

#     # Generate potential errors based on correct prefixes and common OCR errors
#     for prefix in correct_prefixes:
#         for i, correct_char in enumerate(prefix):
#             for wrong_char, right_char in likely_errors.items():
#                 if correct_char == right_char:
#                     # Replace the correct character with a common OCR error in the prefix
#                     incorrect_prefix = list(prefix)
#                     incorrect_prefix[i] = wrong_char
#                     incorrect_prefix = ''.join(incorrect_prefix)

#                     # Add to corrections if it's not already the correct prefix
#                     if incorrect_prefix != prefix:
#                         corrections[incorrect_prefix] = prefix

#     # Apply corrections
#     for incorrect, correct in corrections.items():
#         # Regex to find incorrect prefixes that may be merged with alphanumeric strings
#         pattern = re.escape(incorrect) + r"(?=\d{7})"  # Assuming prefix is followed by 7 digits
#         text = re.sub(pattern, correct, text)

#     return text

# correct_prefixes = ["SNL", "KDD", "YBB"]


def correct_common_ocr_errors(text):
    corrections = {
        r"SNl": "SNL",
        r"SML": "SNL",
        r"5NL": "SNL",
        r"YRT": "YTR",
        r"Y8B": "YBB",
        r"YOB": "YBB",
        r"K00": "KDD",
        r"YB8": "YBB",
        r"KD0": "KDD",
        r"Y88": "YBB"
    }
    for incorrect, correct in corrections.items():
        # Apply corrections even if merged with numbers
        text = re.sub(r"(?<!\w)" + re.escape(incorrect), correct, text)
    return text


# def extract_card_data(ocr_text):
#     name = " "
#     parent_spouse_name = " "
#     house_number = " "
#     age = " "
#     gender = " "

#     name_match = re.search(r"Name\s*:\s*(.+)", ocr_text)
#     parent_spouse_name_match = re.search(r"(Father|Husband)['’`s]* Name\s*:\s*(.+)", ocr_text, re.IGNORECASE)
#     # Adjusted regex for House Number to capture up to the next known fields or end of the line
#     house_number_match = re.search(r"House Number\s*:\s*([^A]+?)(?=\s*Age|\s*Gender|$)", ocr_text)
#     age_match = re.search(r"Age\s*:\s*(\d+)", ocr_text)
#     gender_match = re.search(r"Gender\s*:\s*(Male|Female|Other)", ocr_text, re.IGNORECASE)

#     if name_match and name_match.group(1).strip():
#         name = name_match.group(1).strip()
#     if parent_spouse_name_match and parent_spouse_name_match.group(2).strip():
#         parent_spouse_name = parent_spouse_name_match.group(2).strip()
#     if house_number_match and house_number_match.group(1).strip():
#         house_number = house_number_match.group(1).strip()
#     if age_match and age_match.group(1).strip():
#         age = age_match.group(1).strip()
#     if gender_match and gender_match.group(1).strip():
#         gender = gender_match.group(1).strip()

#     return (name, parent_spouse_name, house_number, age, gender)

# Function to remove punctuation from a string
def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    clean_string = input_string.translate(translator)
    return clean_string

def remove_special_characters(input_string):
    cleaned_string = re.sub(r'[^A-Z0-9]', '', input_string)
    return cleaned_string

def clean_string(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9:\s-]', '', input_string)
    return cleaned_string

#Function to preprocess images for better OCR accuracy
def preprocess_image(image):
    # Step 1: Scale the image
    scale_factor = 3.8  # Increase scale factor for better OCR results
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Step 2: Convert the image to black and white (binary)
    threshold = 180  # Adjust threshold value as needed
    binary_image = image.convert('L').point(lambda p: 255 if p > threshold else 0, mode='1')

    # Step 3: Convert binary image back to "L" mode for further processing
    binary_image = binary_image.convert('L')

    # Step 4: Apply a median filter to reduce noise
    filtered_image = binary_image.filter(ImageFilter.MedianFilter(size=3))

    # Step 5: Enhance sharpness to make the text edges more pronounced
    sharpness_enhancer = ImageEnhance.Sharpness(filtered_image)
    sharpened_image = sharpness_enhancer.enhance(4.0)  # Increase sharpness significantly

    # Step 6: Enhance contrast to make text more distinguishable
    contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
    final_image = contrast_enhancer.enhance(3.0)  # Increase contrast for OCR

    return final_image

def get_pdf_files(folder_path):
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def replace_special_characters_from_marathi(text):
    pattern = r"[^ \u0900-\u097F:\-\d]"
    replaced_text = re.sub(pattern, '', text)
    return replaced_text

def replace_special_characters_from_marathi_words_only(text):
    pattern = r"[^ \u0900-\u097F:\-]"
    replaced_text = re.sub(pattern, '', text)
    return replaced_text

def detect_card_start_end_pages(pdf_path):  # Code for where Page No starts from 4
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    start_page = None
    end_page = None

    for page_num in range(total_pages):
        page = reader.pages[page_num]
        text = page.extract_text()

        # Normalize the text to lower case for easier matching
        text = text.lower()

        # Detect start page
        if not start_page and ("name :" in text and "house number :" in text and "age :" in text and "gender :" in text):
            start_page = page_num + 1  # Adjusting to 1-based indexing

        # Detect end page
        if start_page and ("name :" in text and "house number :" in text and "age :" in text and "gender :" in text):
            end_page = page_num + 1  # Adjusting to 1-based indexing

    # If start_page or end_page are not detected, use fallback values
    if not start_page:
        start_page = 4 if total_pages > 5 else 5  # Fallback to page 4 or 5 depending on total pages
    if not end_page:
        end_page = total_pages - 1 if total_pages > 1 else total_pages - 1  # Fallback to last page -1

    return start_page, end_page

def convert_pdf_to_images(pdf_path, start, end, pdf_filename):
    images = pdf2image.convert_from_path(pdf_path, first_page=start, last_page=end, poppler_path=poppler_path)
    print(f"Image conversion completed for {pdf_filename}")
    return images

def is_image_blank(image, threshold=250, blank_percentage=0.99):
    """
    Checks if the image is mostly blank or white.

    :param image: PIL Image object
    :param threshold: Pixel value threshold to consider a pixel as white (0-255)
    :param blank_percentage: Percentage of pixels that must exceed the threshold to consider the image blank
    :return: True if the image is blank, False otherwise
    """
    # Convert image to grayscale
    grayscale = image.convert('L')
    # Calculate the number of white pixels
    white_pixels = sum(1 for pixel in grayscale.getdata() if pixel > threshold)
    total_pixels = grayscale.width * grayscale.height
    return (white_pixels / total_pixels) > blank_percentage


def clean_data(Name, Parent_name, Home_no):
    try:
        # Ensure input is a string, if not, convert to an empty string
        Name = Name if isinstance(Name, str) else ''
        Parent_name = Parent_name if isinstance(Parent_name, str) else ''
        Home_no = Home_no if isinstance(Home_no, str) else ''

        # Define regex patterns to match based on your logic
        name_pattern = r"^(.*?)(?:\s(?:Husband's Name|Father's Name|Mothers's Name|Others))"
        parent_pattern = r"^(.*?)(?:\s(?:House Number|Husband's Name|Father's Name|Mothers's Name|Others))"
        house_number_pattern = r"^(.*?)(?:\s(?:Age))"

        # Extract name
        name_match = re.search(name_pattern, Name)
        if name_match:
            Name = name_match.group(1).strip()
        else:
            Name = ''  # If pattern does not match, return an empty string

        # Extract parent or spouse name
        parent_match = re.search(parent_pattern, Parent_name)
        if parent_match:
            Parent_name = parent_match.group(1).strip()
        else:
            Parent_name = ''  # If pattern does not match, return an empty string

        # Extract house number
        house_number_match = re.search(house_number_pattern, Home_no)
        if house_number_match:
            Home_no = house_number_match.group(1).strip()
        else:
            Home_no = ''  # If pattern does not match, return an empty string

    except re.error as regex_error:
        # Handle any regex errors, though rare in this case
        print(f"Regex error occurred: {regex_error}")
        return '', '', ''

    except Exception as e:
        # Catch any other unexpected errors
        print(f"An error occurred: {e}")
        return '', '', ''

    # Return the cleaned data
    return Name, Parent_name, Home_no


def process_image(image, page, pdf_filename):
    global sequence_counter  # Use the global sequence_counter to keep it continuous across pages
    card_texts = []
    not_extracted_card_texts = []

    # Preprocess the entire image first
    image = preprocess_image(image)

    # Extract the Assembly Constituency No and Name, Section No and Name, and Part No.
    assembly_constituency_region = image.crop((0, 2.2, image.width // 1.4, image.height // 51))
    section_region = image.crop((0, image.height // 55 + 2.2, image.width // 0.8, (image.height // 63) * 2))  # Increased the width
    part_no_region = image.crop((image.width // 1.2, 1.6, image.width, image.height // 51))

    # Extract text from the regions
    assembly_constituency_text = pytesseract.image_to_string(assembly_constituency_region, lang='eng').strip()
    section_text = pytesseract.image_to_string(section_region, lang='eng').strip()
    part_no_text = pytesseract.image_to_string(part_no_region, lang='eng').strip()

    # Extract only the relevant parts after the colon
    assembly_constituency_extracted = re.search(r'Assembly Constituency No and Name\s*:\s*(.*)', assembly_constituency_text)
    section_extracted = re.search(r'Section No and Name\s*:\s*(.*)', section_text)
    part_no_extracted = re.search(r'Part No\.\s*:\s*(.*)', part_no_text)

    assembly_constituency_final = assembly_constituency_extracted.group(1).strip() if assembly_constituency_extracted else ''
    section_final = section_extracted.group(1).strip() if section_extracted else ''
    part_no_final = part_no_extracted.group(1).strip() if part_no_extracted else ''

    # Extract SublocNo and SublocName using regex
    subloc_match = re.match(r'^(\d+)[-\s]*(.*)', section_final)
    if subloc_match:
        sublocno = subloc_match.group(1).strip()  # Extract the number (SublocNo)
        sublocname = subloc_match.group(2).strip()  # Extract the remaining string (SublocName)
    else:
        sublocno = ''  # If no number is present, leave SublocNo empty
        sublocname = section_final  # Treat the entire section_final as SublocName

    header = image.height // 33
    footer = image.height / 35
    side = image.width // 44
    height = image.height - (header + footer)
    width = image.width - (2.2 * side)

    voter = 0
    prev_number1 = None

    for row in range(10):
        for col in range(3):
            try:
                voter += 1
                x = (col * (width // 3)) + side
                y = (row * (height // 10)) + header
                w = width // 3
                h = height // 10

                card_image = image.crop((x, y, x + w, y + h))
                # plt.imshow(card_image, cmap="gray")
                # plt.axis("off")  # Hide axes
                # plt.show()
                # Check if the image is blank or mostly white
                if is_image_blank(card_image):
                    print(f"Card {voter} is blank or white. Skipping OCR.")
                    continue

                # # Tesseract OCR Processing
                card_text = pytesseract.image_to_string(card_image, lang='eng').strip()
                # print(card_text)
                # If the card_text is empty after OCR, skip the card
                if card_text.strip() == '':
                    continue

                # If card_text contains valid data, increment the sequence counter and assign it
                sequence_counter += 1
                print(f"Card {voter} has valid data. Assigning sequence number: {sequence_counter}")

                # Crop the number1 region
                number_region = card_image.crop((card_image.width * 0.007, card_image.height * 0.02, card_image.width * 0.36, card_image.height * 0.26))

                # Crop the number2 region
                number_region2 = card_image.crop((card_image.width * 0.36, card_image.height * 0.06, card_image.width * 0.58, card_image.height * 0.22))

                # Enhance the image to improve OCR accuracy for number1
                number_region = number_region.convert('L')  # Convert to grayscale
                number_region = number_region.point(lambda p: p > 150 and 255)  # Apply thresholding
                number_region = number_region.filter(ImageFilter.SHARPEN)  # Sharpen the image

                # Enhance the image to improve OCR accuracy for number2
                number_region2 = number_region2.resize((number_region2.width * 2, number_region2.height * 2), Image.Resampling.LANCZOS)
                number_region2 = number_region2.convert('L')
                number_region2 = number_region2.point(lambda p: p > 150 and 255)
                number_region2 = number_region2.filter(ImageFilter.SHARPEN)

                # Perform OCR on the cropped number1 region
                number_text = pytesseract.image_to_string(number_region, lang='eng', config='--psm 7').strip()

                # Perform OCR on the cropped number2 region (use digits-only mode)
                number_text2 = pytesseract.image_to_string(number_region2, config='--psm 7 -c tessedit_char_whitelist=0123456789').strip()

                # Check if the record is deleted (starts with "S")
                deleted = number_text.startswith("S")
                if deleted:
                    Deleted = True  # Set Deleted flag for "S" rows
                else:
                    Deleted = False

                # Use regex to extract the number1 (only digits)
                matches = re.findall(r'\b\d+\b', number_text)

                if matches:
                    number1 = matches[0]
                else:
                    number1 = ""

                # Extract number2 (only digits, should be between 1-10)
                if number_text2.isdigit() and int(number_text2) in range(1, 1111):
                    number2_text = int(number_text2)
                else:
                    number2_text = ""  # Default if not a valid number

                # Handle deleted row with "S" but still process number1
                if deleted:
                    if number1.isdigit():
                        number1_text = int(number1)  # Convert the number for deleted rows
                    else:
                        number1_text = ""  # No valid number extracted
                else:
                    # Regular case (non-deleted rows)
                    if number1.isdigit():
                        number1_text = int(number1)
                    else:
                        number1_text = ""

                # Handle missing number1 based on the previous number
                predicted_number1 = ""
                if not number1_text:
                    if prev_number1 is not None:
                        # Predict number1 using nearby numbers
                        predicted_number1 = prev_number1 + 1
                    else:
                        predicted_number1 = ""

                # If the number1 sequence is broken, use predicted_number1 instead
                if number1_text:
                    # If number1_text is present, check if it matches the prediction, otherwise replace it
                    if prev_number1 and (number1_text != prev_number1 + 1):
                        predicted_number1 = prev_number1 + 1
                    else:
                        predicted_number1 = number1_text  # Use the extracted number if correct
                else:
                    predicted_number1 = prev_number1 + 1 if prev_number1 is not None else ""

                # Update `prev_number1` for future missing number prediction
                prev_number1 = predicted_number1 if isinstance(predicted_number1, int) else None
                

                # Define the crop box dimensions
                # (left, upper, right, lower)
                # Assuming 'image' is the PIL image object loaded with your card image
                left = 1
                upper = card_image.height * 1.7/8  # Start below the header
                right = card_image.width * 2.3/3  # End before the photo
                lower = card_image.height * 7/8  # End above any bottom padding
                
                # Create the crop box
                crop_box = (left, upper, right, lower)
                cropped_image = card_image.crop(crop_box)
                # plt.imshow(cropped_image, cmap="gray")
                # plt.axis("off")  # Hide axes
                # plt.show()

                # # Tesseract OCR Processing
                card_text = pytesseract.image_to_string(cropped_image, lang='eng').strip()
                # name, parent_spouse_name, house_number, age, gender = extract_card_data(card_text)
                # print(f"Extracted Data:\n Name: {name}\n Parent/Spouse Name: {parent_spouse_name}\n House Number: {house_number}\n Age: {age}\n Gender: {gender}")
                # print(card_text)
                # If the card_text is empty after OCR, skip the card
                if card_text.strip() == '':
                    continue
                 
                # Process the card text
                card_text = card_text.strip()
                
                # Standardize the format of key-value pairs by replacing various delimiters with a colon and handling OCR errors
                card_text = card_text.replace('+', ':').replace('=', ':').replace('>', ':') \
                    .replace('*', ':').replace('::', ':').replace('=:', ':') \
                    .replace('Name?', 'Name:').replace('Name!', 'Name:') \
                    .replace('Name >', 'Name:').replace('Name=:', 'Name:') \
                    .replace('Name = :', 'Name :').replace('Name =:', 'Name :')\
                    .replace('Name ! ', 'Name : ').replace("Name ', '", 'Name :')
                # Remove extra spaces around colons
                card_text = ': '.join([part.strip() for part in card_text.split(':')])

                # Additional error handling for unexpected or missing key-value delimiters
                card_text = re.sub(r'Name\s*[:=>\-]+\s*', 'Name: ', card_text)  # Handle cases like "Name =>" or "Name -"

                # Remove any multiple spaces or special characters that shouldn't be there
                card_text = re.sub(r'\s+', ' ', card_text)  # Reduce multiple spaces to one
                card_text = re.sub(r':+', ':', card_text)   # Reduce multiple colons to one
                
                # To enhance clarity and further standardize the card_text, you may also consider cleaning up irregular characters
                card_text = re.sub(r'[\*\$\%\&\@\!\?]', '', card_text)  # Remove special characters that are unlikely to be part of names or addresses
                
                # Print or continue processing with the cleaned-up card text
                # print(f"Cleaned Card Text: {card_text}")

                # Crop the region for Voter ID and EPIC number extraction
                voterid_and_epicno_region = card_image.crop((card_image.width * 0.68, card_image.height * 0.03, card_image.width * 0.995, card_image.height * 0.22))
                # plt.imshow(voterid_and_epicno_region)
                # plt.axis("off")
                # plt.show()
                
                # Convert to grayscale
                grayscale_image = voterid_and_epicno_region.convert("L")
                
                # Enhance sharpness
                sharpness_enhancer = ImageEnhance.Sharpness(grayscale_image)
                sharpened_image = sharpness_enhancer.enhance(3.0)
                
                # Save the preprocessed image to a temporary file
                temp_image_path = "temp_image_for_ocr.png"
                sharpened_image.save(temp_image_path)
                # Increase contrast
                contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
                high_contrast_image = contrast_enhancer.enhance(2.0)
                
                # Apply a filter to reduce noise
                filtered_image = high_contrast_image.filter(ImageFilter.MedianFilter(size=3))
                
                # Optionally enhance brightness
                brightness_enhancer = ImageEnhance.Brightness(filtered_image)
                final_image = brightness_enhancer.enhance(1.2)
                
                # # Save and display the enhanced image for debugging
                final_image.save("enhanced_text_image_pillow.png")
                # plt.imshow(final_image, cmap="gray")
                # plt.axis("off")
                # plt.show()
                

                # Set the logging level to suppress debug information
                logging.getLogger('ppocr').setLevel(logging.ERROR)


                # Initialize PaddleOCR
                ocr = PaddleOCR(use_gpu=False)
                result = ocr.ocr("enhanced_text_image_pillow.png", cls=False)
                
                # Extract text from OCR result and save in `voterid_and_epicno_text`
                voterid_and_epicno_text = ""
                for line in result[0]:  # OCR result is a list of lines
                    voterid_and_epicno_text += line[1][0]  # line[1][0] contains the recognized text

               
                # Save the final extracted text
                voterid_and_epicno_text = voterid_and_epicno_text.strip()  # Clean up leading/trailing spaces
                #Print the final result
                # print(f"Extracted Voter ID: {voterid_and_epicno_text}")



                # # Use Tesseract to extract text with fine-tuned configurations
                # custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                # voterid_and_epicno_text = pytesseract.image_to_string(voterid_and_epicno_region,lang='eng',config=custom_config).strip()
                
                
                # Perform OCR on the cropped region using Tesseract
                # voterid_and_epicno_text = pytesseract.image_to_string(voterid_and_epicno_region, lang='eng', config='--psm 7').strip()

                # Remove spaces from the last 10 characters of the OCR text
                voterid_and_epicno_text = voterid_and_epicno_text[:-10] + voterid_and_epicno_text[-10:].replace(" ", "")
                

                # Apply corrections to the text
                corrected_text = correct_common_ocr_errors(voterid_and_epicno_text)

                # Call the function with the correct_prefixes argument
                # corrected_text = correct_common_ocr_errors(voterid_and_epicno_text, correct_prefixes)

                # Define regular expressions for extracting the Voter ID and EPIC number
                voter_id_pattern = r"\b\d+\b"  # To extract Voter ID (sequence of integers)
                epic_number_pattern = r"\b[A-Z]{3}\d{7}\b"  # To extract EPIC number (3 letters followed by 7 digits)

                # Extract Voter ID (first occurrence of integers in the text)
                voter_id_match = re.search(voter_id_pattern, corrected_text)
                voter_id = voter_id_match.group() if voter_id_match else " "

                # Extract EPIC number (first occurrence of alphanumeric pattern)
                epic_number_match = re.search(epic_number_pattern, corrected_text)
                epic_number = epic_number_match.group() if epic_number_match else " "
                # print("EPIC Number:", epic_number)              
                # Remove any special characters from the EPIC number
                epic_number = re.sub(r'[^A-Za-z0-9]', '', epic_number)

                content = card_text.strip().split(':')
                content1 = card_text.strip().split('-')

                if len(content) > 1 and content[1].strip():
                    Name = content[1].strip().split('-')[0]
                elif len(content1) > 1 and content1[1].strip().split('Name')[1]:
                    Name = content1[1].strip().split('Name')[1].replace(':', '')
                elif len(content1) > 1 and content1[1].strip():
                    Name = content1[1].strip()
                else:
                    Name = ''

                if len(Name) > 2 and not any(char.isdigit() for char in Name):
                    Parent_name = content[2].strip().split('-')[0] if content[2].strip() else None
                else:
                    Name = content[2].strip().split('-')[0] if content[1].strip() else None
                    Parent_name = content[3].strip().split('-')[0] if content[2].strip() else None

                split_result = card_text.strip().split('Number')

                if len(split_result) > 1:
                    house_number_part = split_result[1]
                    house_number_parts = house_number_part.split('-')

                    if len(house_number_parts) > 0:
                        Home_no = house_number_parts[0].replace(':', '').strip()

                        if any(word in Home_no for word in ["Photo", "Pnoto", "Proto", "Available", "Availab!e"]):
                            Home_no = Home_no.replace("Photo", " ") \
                                .replace("Pnoto", " ") \
                                .replace("Proto", " ") \
                                .replace("Available", " ") \
                                .replace("Availab!e", " ") \
                                .strip()
                        else:
                            Home_no = None
                else:
                    Home_no = None

                if 'Female-Available-' in card_text or 'Female-' in card_text:
                    gender = 'FEMALE'
                elif 'Male-Available-' in card_text or 'Male-' in card_text:
                    gender = 'MALE'
                else:
                    gender_pattern = r'Gender[^\w]*(\w+)'
                    gender_matches = re.search(gender_pattern, card_text, re.IGNORECASE)
                    if gender_matches:
                        gender_text = gender_matches.group(1).strip()
                        if gender_text.startswith('M'):
                            gender = 'MALE'
                        elif gender_text.startswith('F'):
                            gender = 'FEMALE'
                        else:
                            gender = ''
                    else:
                        gender = ''

                age_pattern = r':\s(\d{2})\s'
                age_matches = re.findall(age_pattern, card_text)
                age = int(age_matches[0]) if age_matches else None

                Name, Parent_name, Home_no = clean_data(Name, Parent_name, Home_no)
                # print(Name)
                card_texts.append({
                    'PDF_FileName': pdf_filename,
                    'AC_NO_NAME': assembly_constituency_final,
                    'ListNO': part_no_final,
                    'SublocNo': sublocno,
                    'SublocName': sublocname,
                    'Voter_ID': sequence_counter,
                    'EPIC_Number': epic_number,
                    'Name': Name.strip(),
                    'Parents/Spouse/Other_Name': Parent_name.strip(),
                    'Age': age,
                    'House_No': Home_no,
                    'GENDER': gender,
                    'ocr_p_no': page,
                    'ocr_vp_no': voter,
                    'ocr_section_name_no': section_final,
                    'voterid_and_epicno_text': voterid_and_epicno_text,
                    'ocr_card_text': content,
                    'ocr_deleted_flag': Deleted,  # New flag for deleted records
                    'ocr_idcard_no': None,
                    'ocr_number1': number1_text,
                    'ocr_number2': number2_text,
                    'NEW_Voter_ID_2': voter_id,  # OCR extracted Voter ID
                })

            except Exception as e:
                print(f"An error occurred while processing the card: {e}")
                not_extracted_card_texts.append({
                    'pdf_filename': pdf_filename,
                    'P.No': page,
                    "VP.No": voter,
                    'text': str(e)
                })

    return card_texts, not_extracted_card_texts

def process_pdf_files(pdf_path):
    pdf_filename = os.path.basename(pdf_path)

    # Automatically detect start and end pages
    start, end = detect_card_start_end_pages(pdf_path)

    images = convert_pdf_to_images(pdf_path, start=start, end=end, pdf_filename=pdf_filename)
    extracted_text, not_extract_text = [], []

    for page, image in enumerate(images, start=start):
        card_texts, not_extracted_card_texts = process_image(image, page, pdf_filename)
        extracted_text.extend(card_texts)
        not_extract_text.extend(not_extracted_card_texts)

    return extracted_text, not_extract_text

# Function to insert data into the SQL database using ODBC
def insert_into_sqldb_odbc(data, connection_string, table_name, max_retries=5, retry_delay=2):
    retry_count = 0
    conn = None
    while retry_count < max_retries:
        try:
            # Establish the ODBC connection
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()

            # Define the insert query to match the SQL columns
            insert_query = f"""
            INSERT INTO {table_name} (
                ocr_pdf_filename, ocr_ac_name_no, ocr_partno_listno, ocr_subloc_no, ocr_subloc_name, ocr_voter_id,
                ocr_p_no, ocr_vp_no, ocr_epic_number, ocr_name, ocr_parents_spouse_name, ocr_age, ocr_home_no, ocr_gender,
                ocr_section_name_no, voterid_and_epicno_text, ocr_card_text,
                ocr_deleted_flag, ocr_idcard_no, ocr_number1, ocr_number2, ocr_NEW_Voter_ID_2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # Prepare the data as a list of tuples, and ensure None values are handled
            records = [
                (
                    str(row['PDF_FileName']) if row['PDF_FileName'] is not None else None,
                    str(row['AC_NO_NAME']) if row['AC_NO_NAME'] is not None else None,
                    str(row['ListNO']) if row['ListNO'] is not None else None,
                    str(row['SublocNo']) if row['SublocNo'] is not None else None,
                    str(row['SublocName']) if row['SublocName'] is not None else None,
                    str(row['Voter_ID']) if row['Voter_ID'] is not None else None,
                    str(row['ocr_p_no']) if row['ocr_p_no'] is not None else None,
                    str(row['ocr_vp_no']) if row['ocr_vp_no'] is not None else None,
                    str(row['EPIC_Number']) if row['EPIC_Number'] is not None else None,
                    str(row['Name']) if row['Name'] is not None else None,
                    str(row['Parents/Spouse/Other_Name']) if row['Parents/Spouse/Other_Name'] is not None else None,
                    str(row['Age']) if row['Age'] is not None else None,
                    str(row['House_No']) if row['House_No'] is not None else None,
                    str(row['GENDER']) if row['GENDER'] is not None else None,
                    str(row['ocr_section_name_no']) if row['ocr_section_name_no'] is not None else None,
                    str(row['voterid_and_epicno_text']) if row['voterid_and_epicno_text'] is not None else None,
                    str(row['ocr_card_text']) if row['ocr_card_text'] is not None else None,
                    str(row['ocr_deleted_flag']) if row['ocr_deleted_flag'] is not None else None,
                    str(row['ocr_idcard_no']) if row['ocr_idcard_no'] is not None else None,
                    str(row['ocr_number1']) if row['ocr_number1'] is not None else None,
                    str(row['ocr_number2']) if row['ocr_number2'] is not None else None,
                    str(row['NEW_Voter_ID_2']) if row['NEW_Voter_ID_2'] is not None else None
                )
                for _, row in data.iterrows()
            ]

            # Execute the insertion
            cursor.executemany(insert_query, records)

            # Commit the transaction
            conn.commit()

            print(f"Data successfully inserted into {table_name}")
            break  # Break out of the retry loop if successful

        except pyodbc.DatabaseError as e:
            print(f"DatabaseError: {e}")
            retry_count += 1
            print(f"Retrying {retry_count}/{max_retries} after {retry_delay} seconds...")
            time.sleep(retry_delay)  # Wait before retrying

        except Exception as e:
            print(f"Error inserting into database: {e}")
            raise

        finally:
            if conn:
                conn.close()

    if retry_count == max_retries:
        print(f"Failed to insert data after {max_retries} retries.")

def main():
    start_time = time.time()
    print('Starting PDF processing...')

    pdf_files = get_pdf_files(pdf_folder)
    total_pdf = len(pdf_files)
    batch_size = 10

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for batch_start in range(0, total_pdf, batch_size):
            batch_end = min(batch_start + batch_size, total_pdf)
            batch_pdf_files = pdf_files[batch_start:batch_end]
            for pdf_file in batch_pdf_files:
                global sequence_counter  # Declare the use of global variable
                sequence_counter = 0
                pdf_filename = os.path.basename(pdf_file)

                # Process each PDF file
                extracted_text, not_extract_text = process_pdf_files(pdf_file)

                # Create DataFrames for extracted and not extracted data
                df = pd.DataFrame(extracted_text)
                not_df = pd.DataFrame(not_extract_text)

                extracted_dir = os.path.join(output_dir, "Extracted Data Files")
                not_extracted_dir = os.path.join(output_dir, "Not Extracted Data Files")

                for directory in [extracted_dir, not_extracted_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                df_filename = os.path.join(extracted_dir, f"Extracted Data Files ({pdf_filename}).xlsx")
                not_df_filename = os.path.join(not_extracted_dir, f"Not Extracted Data Files ({pdf_filename}).xlsx")

                df.to_excel(df_filename, index=False)
                not_df.to_excel(not_df_filename, index=False)

                print(f"Extracted DataFrame saved: {df_filename}")
                print(f"Not Extracted DataFrame saved: {not_df_filename}")

                # Insert data for this PDF into SQL Server
                if not df.empty:
                    insert_into_sqldb_odbc(df, connection_string, table_name)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken for processing: {elapsed_time:.2f} minutes")

if __name__ == "__main__":
    main()

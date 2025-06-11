"""
File: convert_to_image.py
Author:
Description: Converting .inkml files from the CROHME dataset into readable image files for data preprocessing.
"""

# import libraries
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import zipfile
from tqdm import tqdm
import shutil


# extract the ZIP file data
def extract_zip(zip_path, extract_path):
    """
    :param zip_path: origin path of zip data
    :param extract_path: path where we want to send unzipped data
    :return: Null (just file moving)
    """

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted data to {extract_path}!")


# parse the .inkml content into the strokes and LaTeX annotation from file
def parse_inkml(inkml_path):
    """
    :param inkml_path: path of .inkml file
    :return: the strokes and LaTeX annotation of the .inkml file
    """

    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        # name
        namespace = {'ink': 'http://www.w3.org/2003/InkML'}

        # get strokes
        strokes = []
        for trace in root.findall('.//ink:trace', namespace):
            grid_points = []
            if trace.text:
                coords = trace.text.strip().split(',')
                for coord in coords:
                    parts = coord.strip().split(' ')
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        grid_points.append([x, y])

            if grid_points:
                strokes.append(np.array(grid_points))

        # get LaTeX annotation
        latex = None

        # we have to provide a list of singular symbols to skip in order to make sure the latex annotation is preserved
        single_symbols = ['Closest Strk', 'x', '=', '(', ')', 'a', 'b', 'c', 'd',
                         '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                         '\\prime', '\\alpha', '\\beta', '\\gamma', '\\delta',
                         '\\epsilon', '\\theta', '\\lambda', '\\mu', '\\pi', '\\sigma',
                         '\\phi', '\\psi', '\\omega', '\\sum', '\\int', '\\infty',
                         '\\sqrt', '\\frac', '\\cdot', '\\times', '\\div', '\\leq',
                         '\\geq', '\\neq', '\\approx', '\\in', '\\subset', '\\cup',
                         '\\cap', '\\forall', '\\exists', '\\partial', '\\nabla']

        for annotation in root.findall('.//ink:annotation', namespace):
            if annotation.text and annotation.get('type') == 'truth':
                txt = annotation.text.strip()
                if txt and txt not in single_symbols and len(txt) > 2:
                    if txt.startswith('$') and txt.endswith('$'):
                        latex = txt[1:-1]
                    else:
                        latex = txt
                    break
        return strokes, latex

    except ET.ParseError as e:
        # XML parsing error (common in artificial data)
        if "Artificial_data" in inkml_path:
            # skip artificial data errors
            return None, None
        else:
            print(f"XML Parse Error in {inkml_path}: {e}")
            return None, None

    except Exception as e:
        print(f"Error parsing {inkml_path}: {e}")
        return None, None


# normalize the strokes to fit within the desired target size
def normalize_strokes(curr_strokes, desired_target_size=(224, 224), padding=16):
    """
    :param curr_strokes: the strokes of the file we're looking at
    :param desired_target_size: the desired target size with padding
    :param padding: size of padding
    :return: the normalized strokes
    """
    if not curr_strokes:  # if we have no strokes
        return curr_strokes

    # creating a bounding box
    points = np.concatenate(curr_strokes)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # now find dimensions of bounding box
    height = max_y - min_y
    width = max_x - min_x

    if height == 0 or width == 0:  # if one dimension has no variation
        return curr_strokes

    # scale and center the strokes
    scale = min((desired_target_size[0] - 2 * padding) / width,
                (desired_target_size[1] - 2 * padding) / height)
    center_x, center_y = desired_target_size[0] / 2, desired_target_size[1] / 2
    stroke_center_x, stroke_center_y = np.mean([min_x, max_x]), np.mean([min_y, max_y])

    # normalize each stroke
    norm_strokes = []
    for stroke in curr_strokes:
        norm = np.zeros_like(stroke)
        norm[:, 0] = (stroke[:, 0] - stroke_center_x) * scale + center_x
        norm[:, 1] = (stroke[:, 1] - stroke_center_y) * scale + center_y
        norm_strokes.append(norm)
    return norm_strokes


# converting the strokes into PIL images
def stroke_to_image(curr_strokes, img_size=(224, 224), line_width=2):
    """
    Render strokes directly to a PIL image using ImageDraw (no matplotlib).
    :param curr_strokes: list of (N, 2) numpy arrays (already normalized to image space)
    :param img_size: output image size
    :param line_width: thickness of lines
    :return: PIL.Image
    """
    img = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(img)

    for stroke in curr_strokes:
        if len(stroke) > 1:
            points = [tuple(p) for p in stroke]
            draw.line(points, fill='black', width=line_width, joint='curve')

    return img


# convert an .inkml file to .png
def inkml_to_png(inkml_path, png_path, img_size=(224, 224)):
    """
    :param inkml_path: input path of .inkml
    :param png_path: output path of .png
    :param img_size: image size we want
    :return: if we had strokes, and the latex annotation
    """

    # get strokes and latex annotation
    strokes, latex = parse_inkml(inkml_path)

    if not strokes:
        return False, None

    # normalize strokes
    norm_strokes = normalize_strokes(strokes, desired_target_size=img_size)

    # convert strokes to image
    image = stroke_to_image(norm_strokes, img_size=img_size)

    # create output path for png image
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    # save image
    image.save(png_path, 'PNG')

    return True, latex


# find all the .inkml in a directory
def find_inkml_files(dir):
    """
    :param dir: the directory path
    :return: all .inkml files found
    """

    # gather .inkml files
    inkml_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.inkml'):
                inkml_files.append(os.path.join(root, file))
    return inkml_files


# convert all the .inkml files into .png files
def convert_dataset(input_dir, output_dir, img_size=(224, 224)):
    """
    :param input_dir: the specified input directory for the files
    :param output_dir: the specified output directory for the files
    :param img_size: the image size we want
    :return: Null (performing task of converting dataset files)
    """

    print(f"Searching for InkML files in {input_dir}...")
    inkml_files = find_inkml_files(input_dir)
    print(f"Found {len(inkml_files)} InkML files")

    if not inkml_files:  # if there's no available .inkml files
        print("No InkML files found!")
        return

    # create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # keep track of file conversion successes
    successes = 0
    failures = 0
    artificial_failures = 0
    absent_latex_count = 0
    labels = []

    # look through all .inkml file paths
    for inkml_path in tqdm(inkml_files, desc="Converting"):

        # create output path
        rel_path = os.path.relpath(inkml_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        output_path = output_path.replace('.inkml', '.png')

        # convert file
        success, latex = inkml_to_png(inkml_path, output_path, img_size)

        if success:
            successes += 1
            if latex:
                labels.append(f"{os.path.basename(output_path)}\t{latex}")
            else:
                absent_latex_count += 1
        else:
            failures += 1
            if 'Artificial_data' in inkml_path:
                artificial_failures += 1

    # save the labels file
    if labels:
        labels_path = os.path.join(output_dir, 'labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
        print(f"\nSaved {len(labels)} LaTeX labels to {labels_path}")
    else:
        print("\nNo LaTeX annotations found in any files!")

    print(f"\nConversion complete!")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures} (including {artificial_failures} artificial data files)")
    print(f"  Files without LaTeX: {absent_latex_count}")
    print(f"  Files with LaTeX: {len(labels)}")
    print(f"  Success rate: {successes/(successes+failures)*100:.1f}%")
    print(f"  Output directory: {output_dir}")


def main():

    # initialize
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    zip_path = os.path.join(base_dir, 'data', 'CROHME23.zip')
    temp_dir = os.path.join(base_dir, 'data', 'extracted_crohme_zip_file')
    output_dir = os.path.join(base_dir, 'data', 'crohme_images')
    img_size = (224, 224)
    # keep_temp_files = True  # we only set to True in the case of debugging

    # check if the zip file exists:
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found!")
        return

    # try to convert .inkml files to images
    try:

        # extract zip file
        extract_zip(zip_path, temp_dir)

        # convert .inkml files
        convert_dataset(temp_dir, output_dir, img_size)
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary files")


if __name__ == '__main__':
    main()

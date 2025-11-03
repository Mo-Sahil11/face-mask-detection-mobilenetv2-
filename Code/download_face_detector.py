#!/usr/bin/env python3
"""
Script to download the face detector model files required for mask detection.
This downloads the OpenCV DNN face detector model (ResNet-10 SSD).
"""

import os
import urllib.request
import urllib.error

# URLs for the face detector model files
FACE_DETECTOR_URLS = {
    'deploy.prototxt': 'https://github.com/opencv/opencv/raw/master/samples/dnn/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
}

def download_file(url, filepath):
    """Download a file from URL to filepath."""
    print(f"[INFO] Downloading {os.path.basename(filepath)}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[INFO] Successfully downloaded {os.path.basename(filepath)}")
        return True
    except urllib.error.URLError as e:
        print(f"[ERROR] Failed to download {os.path.basename(filepath)}: {e}")
        return False

def main():
    # Create face_detector directory if it doesn't exist
    face_detector_dir = os.path.join(os.path.dirname(__file__), 'face_detector')
    os.makedirs(face_detector_dir, exist_ok=True)
    
    # Download each file
    success = True
    for filename, url in FACE_DETECTOR_URLS.items():
        filepath = os.path.join(face_detector_dir, filename)
        if os.path.exists(filepath):
            print(f"[INFO] {filename} already exists, skipping download.")
        else:
            if not download_file(url, filepath):
                success = False
    
    if success:
        print("\n[INFO] All face detector model files downloaded successfully!")
        print(f"[INFO] Files are located in: {face_detector_dir}")
    else:
        print("\n[ERROR] Some files failed to download. Please download manually:")
        print("\nManual download instructions:")
        print("1. Create a directory named 'face_detector' in the Code folder")
        print("2. Download the following files:")
        for filename, url in FACE_DETECTOR_URLS.items():
            print(f"   - {filename}: {url}")

if __name__ == "__main__":
    main()


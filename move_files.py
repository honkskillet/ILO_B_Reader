import os, shutil
origFolder = "/Users/white/Downloads/NIH_CXR_Images/"
destFolder = "/Users/white/Documents/AlexProjects/DICOM_OCC_MED/dicom/NIH_images/"

image_folders = os.listdir(origFolder)
for folder in image_folders:
    if not folder.startswith('.'):
        print(f"moving images in {folder}")
        files = os.listdir(f"{origFolder}{folder}")
        print(f"moving {len(files)} images")
        files.sort()
        for f in files:
            if not f.startswith('.'):
                print('.', end = "")
                src = f"{origFolder}{folder}/{f}"
                dst = f"{destFolder}{f}"
                shutil.move(src,dst)

    
import os

# Check the existence of following folders [datasets, images, input, outputs]
folders = ["datasets", "images", "input", "outputs"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
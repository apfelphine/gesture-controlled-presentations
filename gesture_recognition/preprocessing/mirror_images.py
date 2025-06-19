import os
from PIL import Image


def mirror_images_in_directory(parent_dir):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                if '_mirrored' in file:
                    continue  # Skip already mirrored images

                full_path = os.path.join(root, file)

                try:
                    # Open image
                    img = Image.open(full_path)

                    # Mirror (left-right flip)
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    # Construct new filename
                    filename, ext = os.path.splitext(file)
                    mirrored_filename = f"{filename}_mirrored{ext}"
                    mirrored_path = os.path.join(root, mirrored_filename)

                    # Save mirrored image
                    mirrored_img.save(mirrored_path)
                    print(f"Mirrored and saved: {mirrored_path}")

                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")


if __name__ == "__main__":
    directory_to_process = "../../data/training"
    mirror_images_in_directory(directory_to_process)

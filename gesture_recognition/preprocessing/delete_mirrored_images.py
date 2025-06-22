import os


def delete_mirrored_images(parent_dir):
    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith(image_extensions) and '_mirrored' in file:
                mirrored_path = os.path.join(root, file)
                try:
                    os.remove(mirrored_path)
                    print(f"Deleted mirrored image: {mirrored_path}")
                except Exception as e:
                    print(f"Failed to delete {mirrored_path}: {e}")


if __name__ == "__main__":
    directory_to_process = "../../data/training"
    delete_mirrored_images(directory_to_process)

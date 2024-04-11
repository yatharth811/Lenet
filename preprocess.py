import cv2
import numpy as np
import os

np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})
def process_images(input_folder, output_folder, output_list_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    generated_files = []  # List to store generated file names

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"): # Process only PNG files
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, 0)
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28))
            img = img.reshape(28, 28, -1)

            # Revert the image and normalize it to 0-1 range
            img = img / 255.0

            flattened_img = np.ravel(img)
            output_filename = os.path.join(output_folder,filename[:-4])
            np.savetxt(output_filename, flattened_img, delimiter=",", fmt="%0.6f")
            generated_files.append(filename[:-4])  # Add generated filename to the list
            print(f"Processed {filename} and saved the output to {output_filename}")

    # Save the list of generated files to a separate file
    with open(output_list_file, 'w') as file:
        for generated_file in generated_files:
            file.write(f"{generated_file}\n")
    print(f"Generated file list saved to {output_list_file}")


if __name__ == "__main__":

    input_folder = "img"
    output_folder = "pre-proc-img"
    output_list_file = "img_path.txt"

    process_images(input_folder, output_folder, output_list_file)

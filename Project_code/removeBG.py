import os
import rembg

input_folder = '/Users/zhuhongyun/PycharmProjects/pythonProject1/Project/data/zalando/sweatshirt-female'
output_folder = '/Users/zhuhongyun/PycharmProjects/pythonProject1/Project/data/withoutBG/sweatshirt-female'

# Get the filenames of all files in a folder
file_names = os.listdir(input_folder)

for file_name in file_names:
    if file_name.endswith('.png') or file_name.endswith('.jpg'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, 'bg_removed_' + file_name)
        with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
            # Use the rembg library to remove the background
            output_file.write(rembg.remove(input_file.read()))

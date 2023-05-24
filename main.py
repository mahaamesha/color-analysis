import os

import mafunction as f
from mafunction import input_folder_path



if __name__ == '__main__':
    for fname in os.listdir(input_folder_path)[1:]:
        file_path = os.path.join(input_folder_path, fname)
    
        print(file_path)
        im_roi = f.im_process(file_path)
        
        # background_color = f.detect_background_color(file_path)
        # print(f"Background Color (BGR): {background_color}")
        # break
        
import os

import mafunction as f
from mafunction import input_folder_path



if __name__ == '__main__':
    for fname in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, fname)
    
        print(file_path)
        im_roi = f.im_process(file_path)
        fig_hist = f.get_rgb_histogram(im_roi)
        f.savefig_plt(fig_hist, file_path, sufix='_hist.jpg')
        
        
        # background_color = f.detect_background_color(file_path)
        # print(f"Background Color (BGR): {background_color}")
        # break
        
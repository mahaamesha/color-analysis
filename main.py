import os

import mafunction as f
from mafunction import input_folder_path
from paralleling import do_paralleling


def processing_func(file_path):
    print('    Processing %s ...' %os.path.basename(file_path), end=' ')
    im_roi = f.im_process(file_path)
    fig_hist = f.get_rgb_histogram(im_roi)
    f.savefig_plt(fig_hist, file_path, sufix='_hist.jpg')
    print('OK')


if __name__ == '__main__':
    print('Running pine resin color analysis ...')
    file_paths = [ os.path.join(input_folder_path, fname) for fname in os.listdir(input_folder_path)]
    _, elapsed_time = do_paralleling(file_paths, processing_func)
    print(f'DONE ({elapsed_time} seconds)')
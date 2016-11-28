import argparse
import os

import shutil
from zipfile import ZipFile

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-o', '--out', default='save.zip')
    args = parser.parse_args()
    return args


def zip_save(args):
    temp_dir = "."
    save_dir = os.path.join(temp_dir, "save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for save_source_path in tqdm(args.paths):
        # path = "out/basic/30/save/basic-18000"
        # target_path = "save_dir/30/save"
        # also output full path name to "save_dir/30/readme.txt
        # need to also extract "out/basic/30/shared.json"
        temp, _ = os.path.split(save_source_path)  # "out/basic/30/save", _
        model_dir, _ = os.path.split(temp)  # "out/basic/30, _
        _, model_name = os.path.split(model_dir)
        cur_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        save_target_path = os.path.join(cur_dir, "save")
        shared_target_path = os.path.join(cur_dir, "shared.json")
        readme_path = os.path.join(cur_dir, "readme.txt")
        shared_source_path = os.path.join(model_dir, "shared.json")
        shutil.copy(save_source_path, save_target_path)
        shutil.copy(shared_source_path, shared_target_path)
        with open(readme_path, 'w') as fh:
            fh.write(save_source_path)

    os.system("zip {} -r {}".format(args.out, save_dir))

def main():
    args = get_args()
    zip_save(args)

if __name__ == "__main__":
    main()

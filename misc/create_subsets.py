import os
from pathlib import Path

ilsvrc_root = os.environ.get("ILSVRC_ROOT")
assert ilsvrc_root and os.path.isdir(os.path.expanduser(ilsvrc_root)), "ILSVRC_ROOT is not set or does not exist"
subsets = {
    # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    "felidae": list(range(281, 294)),
    "vessels": [756, 899, 900, 901,
                440, 720, 737, 898,
                907, 504, 896, 438,
                883, 469, 505, 849,
                427, 435, 463, 618,
                666, 725, 876],
    "buildings": [663, 698, 410, 449, 832, 497, 668, 425, 498, 598, 580, 624, 727, 762],
    "nutriment": [928, 929, 927, 923, 925, 926, 965,  962, 963, 964],

}

dst_root = Path("../resources/datasets")
ilsvrc_root = Path(ilsvrc_root)

train_dir = os.listdir(ilsvrc_root / "train")
val_dir = os.listdir(ilsvrc_root / "val")
train_dir.sort(), val_dir.sort()

for subset, class_idxs in subsets.items():
    dst_subset = dst_root / subset
    dst_train = dst_subset / "train"
    dst_val = dst_subset / "val"
    for class_idx in class_idxs:
        class_dir_name = train_dir[class_idx]
        assert class_dir_name == val_dir[class_idx]
        src_train = ilsvrc_root / "train" / class_dir_name
        src_val = ilsvrc_root / "val" / class_dir_name
        os.makedirs(dst_train, exist_ok=True)
        os.makedirs(dst_val, exist_ok=True)
        os.system(f"cp -r {src_train} {dst_train}")
        os.system(f"cp -r {src_val} {dst_val}")

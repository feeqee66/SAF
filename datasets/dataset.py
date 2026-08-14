from pathlib import Path
import numpy as np


class SAFDataset:

    def __init__(self, root):

        self.root = Path(root)

        self.lr_dir = self.root / "NoisyLR"
        self.gt_dir = self.root / "GT"

        self.lr_files = sorted(self.lr_dir.glob("*.npy"))

        self.pairs = []

        for lr_file in self.lr_files:

            gt_file = self.gt_dir / lr_file.name

            if gt_file.exists():
                self.pairs.append((lr_file, gt_file))

        print("Dataset loaded successfully")
        print("Number of pairs:", len(self.pairs))


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, index):

        lr_path, gt_path = self.pairs[index]

        lr = np.load(lr_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        return {
            "name": lr_path.name,
            "lr": lr,
            "gt": gt
        }


if __name__ == "__main__":

    dataset = SAFDataset(
        r"C:\Users\apurva vivobook\Downloads\train\train\train"
    )

    sample = dataset[0]

    print()
    print("Filename:", sample["name"])
    print("LR shape:", sample["lr"].shape)
    print("GT shape:", sample["gt"].shape)
    print("LR dtype:", sample["lr"].dtype)
    print("GT dtype:", sample["gt"].dtype)
    print("LR range:", sample["lr"].min(), sample["lr"].max())
    print("GT range:", sample["gt"].min(), sample["gt"].max())
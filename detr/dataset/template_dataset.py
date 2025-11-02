from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class TemplateDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.files[idx]
        return data

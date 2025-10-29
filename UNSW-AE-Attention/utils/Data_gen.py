import torch
from torch.utils.data import Dataset

class Data_gen(Dataset):
    """
    UNSW-NB15 verisini PyTorch DataLoader ile kullanılabilir hale getirir.
    """
    def __init__(self, data):
        super(Data_gen, self).__init__()
        # DataFrame -> Tensor (float32)
        self.data = torch.tensor(data.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

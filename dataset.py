import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random

class HR_IMG(Dataset):
    def __init__(self, dataset_path, crop_size, norm_mean=0, norm_std=1):
        names = os.listdir(dataset_path)
        self.paths = [os.path.join(dataset_path, x) for x in names]
        
        #self.normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        self.normalize = lambda x: (x - 128)/128
        self.denormalize = lambda x: x*128 + 128

        self.crop = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.ToTensor()
         ])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]

        #im = np.array(Image.open(path).convert('RGB'))
        im = Image.open(path).convert('RGB')
        x = self.crop(im)
        #data = torch.from_numpy(np.float32(im))
        #x = x.permute(2, 0, 1)
        #x = self.normalize(x)
        return x

    def random_crop(self, x):
        w = random.randint(0, x.shape[0] - self.crop_size)
        h = random.randint(0, x.shape[1] - self.crop_size)
        x = x[w:w+s, h:h+s]
        return x



if __name__ == "__main__":
    path = "/data/ai-datasets/202-DIV2K/High-Resolution/DIV2K_train_HR"
    dset = HR_IMG(path)
    #for i in range(len(dset)):
    zero = dset[30]
    print(zero)
    print(zero.size())


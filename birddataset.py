class BirdDataset(Dataset):
    def __init__(self, h5_file, train=True, transform=None):
        self.h5_file = h5_file
        self.train = train
        self.transform = transform
        
        with h5py.File(h5_file, 'r') as hf:
            self.species = list(hf.attrs['species'])
            if train:
                self.length = len(hf['X_train'])
            else:
                self.length = len(hf['X_test'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            if self.train:
                img = hf['X_train'][idx]
                label = hf['y_train'][idx]
            else:
                img = hf['X_test'][idx]
                label = hf['y_test'][idx]
        
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

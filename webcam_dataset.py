from torch.utils.data.dataset import Dataset

class WebcamDataset(Dataset):
    def __init__(self, data, height, width, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Create an empty numpy array to fill
        img_as_np = np.ones((28, 28), dtype='uint8')
        # Fill the numpy array with data from pandas df
        for i in range(1, self.data.shape[1]):
            row_pos = (i - 1) // self.height
            col_pos = (i - 1) % self.width
            img_as_np[row_pos][col_pos] = self.data.iloc[index][i]
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)
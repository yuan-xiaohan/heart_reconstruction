import torch
from torch.utils.data.dataset import Dataset
from utils import *
RESIZE_NUM = 40
Z_SLICE = 336


# png to 3D mat
def png2mat(dir_list):
    num = len(dir_list)
    matrix = np.zeros((RESIZE_NUM, RESIZE_NUM, num), dtype=np.uint8)
    n = 0
    for path in dir_list:
        img = cv2.resize(cv2.imread(path, 0), (RESIZE_NUM, RESIZE_NUM))
        matrix[:, :, n] = img
        # matrix[:, :, n] = cv2.imread(path)[:, :, 1]
        n = n + 1
    return matrix


class DataSet(Dataset):
    def __init__(self, model_path, pca_dir):
        model_list = []
        for root, dirs, files in os.walk(model_path):
            if dirs == []:
                if os.path.split(root)[1] == "img":
                    # print(root)
                    model_list.append(root)
        self.model_list = model_list
        self.pca = np.loadtxt(pca_dir, dtype=np.float)
        # Calculate len
        self.data_len = len(self.model_list)


    def __getitem__(self, index):
        """
        # GET VOLUME
        """
        dir_list = glob.glob(self.model_list[index] + "\*")
        # png2mat
        image_data = png2mat(dir_list)

        # get volume
        image_data = windowAdjust(image_data, 800, 200)
        image_data = norm_img(image_data)  # Normalize the image

        # expand to 256*256*350
        volume = np.zeros([RESIZE_NUM, RESIZE_NUM, Z_SLICE])
        volume[:, :, :image_data.shape[2]] = image_data
        volume[:, :, image_data.shape[2]:] = np.zeros([RESIZE_NUM, RESIZE_NUM, Z_SLICE-image_data.shape[2]])

        volume = np.expand_dims(volume, axis=0)  # add additional dimension [1,h,w,c]
        image_tensor = torch.from_numpy(volume).float()

        """
        # GET PCA
        """
        pca_data = self.pca.T[index]
        pca_tensor = torch.from_numpy(pca_data)  # Convert numpy array to tensor

        return image_tensor, pca_tensor


    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    input_path = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\img"
    pca_dir = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\pca_coefficient.txt"
    dataset = DataSet(input_path, pca_dir)
    pca = np.loadtxt(pca_dir, dtype=np.float)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, batch_size=8, shuffle=True)

    image_tensor, pca_tensor = dataset.__getitem__(0)
    print(image_tensor.shape)
    print(pca_tensor.shape)

from dataset import *
import torch.nn as nn
from Unet3D import UNet3D_Encoder
from PCA_obj import *

# train
def train(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    total_train_loss = 0
    for batch, (volume, label) in enumerate(train_loader):
        volume = volume.cuda()
        label = label.cuda()
        output = model(volume)

        loss = criterion(output, label.float())  # 不同损失对label要求不同
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


        # print(
        #     'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tdice: {}'.format(
        #         epoch, batch, len(train_loader), 100. * batch / len(train_loader), train_loss.item(), train_dice))
        # total_train_loss = total_train_loss + loss.cpu().item()
        total_train_loss += loss.cpu().item()

    return total_train_loss/(batch + 1)


def val(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0
    for batch, (volume, label) in enumerate(val_loader):
        volume = volume.cuda()
        label = label.cuda()
        output = model(volume)

        loss = criterion(output, label.float())  # 不同损失对label要求不同
        total_val_loss = total_val_loss + loss.cpu().item()

    return total_val_loss/(batch + 1)



def main():
    input_path_train = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\train\img"
    pca_dir_train = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\train\pca_coefficient.txt"
    save_path = r"D:\XiaohanYuan\heart_reconstruction\3D\history"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # input_path_train = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\test\img"
    # pca_dir_train = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\test\pca_coefficient.txt"
    # save_path = r"D:\XiaohanYuan\heart_reconstruction\3D\history"

    pca_dir_all = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\pca_coefficient_all.txt"
    input_path_val = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\val\img"
    pca_dir_val = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\val\pca_coefficient.txt"


    dataset_train = DataSet(input_path_train, pca_dir_train)
    dataset_val = DataSet(input_path_val, pca_dir_val)
    pca_all = np.loadtxt(pca_dir_all, dtype=np.float)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False)
    model = UNet3D_Encoder(1, [32, 64, 128, 256, 512], pca_all.shape[0], net_mode='3d')
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

    model.load_state_dict(torch.load(r"D:\XiaohanYuan\heart_reconstruction\3D\model_epoch_1000.pkl"))

    # torch.backends.cudnn.enabled = False  # ???
    criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(model.module.parameters(),
                                 lr=1e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[20, 50],
                                                     gamma=0.9)

    epoch_start = 0
    epoch_end = 10000
    # Train
    print("Initializing Training!")
    for epoch in range(epoch_start, epoch_end):
        total_train_loss = train(model, train_loader, criterion, optimizer, scheduler)
        print('Epoch', str(epoch + 1), 'Train loss:', total_train_loss)
        if (epoch + 1) % 10 == 0:
            total_val_loss = val(model, val_loader, criterion)
            print('Epoch', str(epoch + 1), 'Val loss:', total_val_loss)
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), save_path + r"\model_epoch_{0}.pkl".format(epoch+1))

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import thop
from sklearn.metrics import classification_report
import os
import randomSampler
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net, Net_one
import random
from dataset2 import getDataset
from utils import parse_acc_from_classifaction_report
from center_loss import CenterLoss

max_epoch = 250
img_size = 224
batch_size = 128
f1_list = []
sq = 'all'

def init_seed(seed):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def train():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],
                             [0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset1 = getDataset(path='CBF', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset1 = getDataset(path='CBF', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset2 = getDataset(path='CBV', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset2 = getDataset(path='CBV', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset3 = getDataset(path='T1', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset3 = getDataset(path='T1', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset4 = getDataset(path='T1c', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset4 = getDataset(path='T1c', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset5 = getDataset(path='Flair', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset5 = getDataset(path='Flair', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset6 = getDataset(path='T2', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset6 = getDataset(path='T2', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset7 = getDataset(path='rMD', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset7 = getDataset(path='rMD', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_dataset8 = getDataset(path='rFA', img_size=img_size,
                                transform=train_transforms, is_training=True)
    val_dataset8 = getDataset(path='rFA', img_size=img_size,
                              transform=val_transforms, is_training=False)

    train_s = torch.randperm(len(train_dataset1)).tolist()
    train_sampler = randomSampler.RandomSampler(train_dataset1, train_s)
    val_s = torch.randperm(len(val_dataset1)).tolist()
    val_sampler = randomSampler.RandomSampler(val_dataset1, val_s)

    train_dataloader1 = DataLoader(train_dataset1, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader2 = DataLoader(train_dataset2, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader3 = DataLoader(train_dataset3, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader3 = DataLoader(val_dataset3, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader4 = DataLoader(train_dataset4, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader4 = DataLoader(val_dataset4, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader5 = DataLoader(train_dataset5, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader5 = DataLoader(val_dataset5, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader6 = DataLoader(train_dataset6, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader6 = DataLoader(val_dataset6, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader7 = DataLoader(train_dataset7, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader7 = DataLoader(val_dataset7, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)
    train_dataloader8 = DataLoader(train_dataset8, batch_size=batch_size, num_workers=0,
                                   drop_last=False, sampler=train_sampler, pin_memory=True)
    val_dataloader8 = DataLoader(val_dataset8, batch_size=1, shuffle=False, num_workers=0,
                                 drop_last=False, sampler=val_sampler, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelName = str(img_size) + 'x' + str(img_size)
    model = Net(num_sq=8).to(device)
    x = torch.randn(32, 1, img_size, img_size).cuda()
    flops, params = thop.profile(model.to("cuda"), inputs=(x, x, x, x, x, x, x, x))
    print("  %s   | %s | %s" % ("Model", "Params(Mb)", "FLOPs(Mb)"))
    print("%s |    %.2f    | %.2f" % (modelName, params / (1024 ** 2), flops / (1024 ** 2)))

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.0005,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0.0001,
                                 amsgrad=False
                                 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.99)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = CenterLoss(num_classes=2, feat_dim=256)

    best_f1 = 0
    for epoch in range(0, max_epoch):
        best_model = model
        model.train()
        loss = None
        train_acc = []
        all_train = zip(enumerate(train_dataloader1),
                        enumerate(train_dataloader2),
                        enumerate(train_dataloader3),
                        enumerate(train_dataloader4),
                        enumerate(train_dataloader5),
                        enumerate(train_dataloader6),
                        enumerate(train_dataloader7),
                        enumerate(train_dataloader8))
        for (i, (imgs1, labels1)), (_, (imgs2, labels2)), (_, (imgs3, labels3)), (_, (imgs4, labels4)), \
            (_, (imgs5, labels5)), (_, (imgs6, labels6)), (_, (imgs7, labels7)), (_, (imgs8, labels8)) \
                in all_train:

            x1 = imgs1.to(device)
            x2 = imgs2.to(device)
            x3 = imgs3.to(device)
            x4 = imgs4.to(device)
            x5 = imgs5.to(device)
            x6 = imgs6.to(device)
            x7 = imgs7.to(device)
            x8 = imgs8.to(device)

            y = labels1.to(device)
            feat, pre_y = model(x1, x2, x3, x4, x5, x6, x7, x8)
            loss1 = criterion1(pre_y, y)
            loss2 = criterion2(feat, y)
            if i % 100 == 0:
                print(epoch, (loss1.item()+loss2.item()))
            optimizer.zero_grad()
            (loss1+loss2).backward()
            optimizer.step()
            ret, predictions = torch.max(pre_y.data, 1)
            correct_counts = predictions.eq(y.data.view_as(predictions))
            tacc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc.append(tacc)
        scheduler.step()

        # 验证
        if (epoch + 1) % 1 == 0:
            model.eval()
            pred_list = []
            label_list = []
            all_val = zip(enumerate(val_dataloader1),
                          enumerate(val_dataloader2),
                          enumerate(val_dataloader3),
                          enumerate(val_dataloader4),
                          enumerate(val_dataloader5),
                          enumerate(val_dataloader6),
                          enumerate(val_dataloader7),
                          enumerate(val_dataloader8))
            for (i, (imgs1, labels1)), (_, (imgs2, labels2)), (_, (imgs3, labels3)), (_, (imgs4, labels4)), \
                (_, (imgs5, labels5)), (_, (imgs6, labels6)), (_, (imgs7, labels7)), (_, (imgs8, labels8)) \
                    in all_val:

                x1 = imgs1.to(device)
                x2 = imgs2.to(device)
                x3 = imgs3.to(device)
                x4 = imgs4.to(device)
                x5 = imgs5.to(device)
                x6 = imgs6.to(device)
                x7 = imgs7.to(device)
                x8 = imgs8.to(device)

                y = labels1.to(device).item()
                _, pre = model(x1, x2, x3, x4, x5, x6, x7, x8)
                pred_cls = torch.argmax(pre).item()
                pred_list.append(pred_cls)
                label_list.append(y)
            report = classification_report(label_list, pred_list, labels=[0, 1],
                                           target_names=['SBM', 'GBM'])
            f1, vacc = parse_acc_from_classifaction_report(report)
            f1_list.append(f1)
            print('train acc:{:.4f}, val acc:{:.4f}'.format((sum(train_acc) / len(train_acc)),vacc))
            print(f'f1:{f1}')
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                print(report)
        if epoch == max_epoch - 1:
            torch.save(model.state_dict(), f'./trained_models/{sq}_{img_size}_last.pth')
            torch.save(best_model.state_dict(), f'./trained_models/{sq}_{img_size}_best.pth')
        df = pd.DataFrame(data=f1_list)
        df.to_csv(f'./trained_models/{sq}_{img_size}.csv',
                  mode="a", encoding="utf_8_sig")


if __name__ == "__main__":
    init_seed(1)
    train()
    os.system("shutdown -s -t  60")

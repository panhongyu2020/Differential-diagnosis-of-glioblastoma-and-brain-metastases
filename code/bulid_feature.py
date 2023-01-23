import pandas as pd
import torch
from model import Net,Net2
from dataset2 import getDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


model = Net2()

model.load_state_dict(torch.load('./trained_models/all2_224.pth',
                                map_location='cuda:0'), strict=False)



if __name__ == '__main__':
    feature_list = []
    img_size = 224
    is_training = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        model.cuda()
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset1 = getDataset(path='CBF', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset2 = getDataset(path='CBV', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset3 = getDataset(path='T1', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset4 = getDataset(path='T1c', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset5 = getDataset(path='Flair', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset6 = getDataset(path='T2', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        dataset7 = getDataset(path='rMD', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)
        dataset8 = getDataset(path='rFA', img_size=img_size,
                                  transform=val_transforms, is_training=is_training)

        # s = torch.randperm(len(dataset1)).tolist()
        # sampler = randomSampler.RandomSampler(dataset1, s)
        person_list = dataset8.person_name

        dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader2 = DataLoader(dataset2, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader3 = DataLoader(dataset3, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader4 = DataLoader(dataset4, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader5 = DataLoader(dataset5, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader6 = DataLoader(dataset6, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader7 = DataLoader(dataset7, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        dataloader8 = DataLoader(dataset8, batch_size=1, num_workers=0,
                                     drop_last=False, pin_memory=True)
        all_val = zip(enumerate(dataloader1),
                      enumerate(dataloader2),
                      enumerate(dataloader3),
                      enumerate(dataloader4),
                      enumerate(dataloader5),
                      enumerate(dataloader6),
                      enumerate(dataloader7),
                      enumerate(dataloader8))
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

            pre_y = model(x1, x2, x3, x4, x5, x6, x7, x8)
            p = (pre_y.tolist())[0]
            l = (labels1.tolist())[0]
            p.insert(0, l)
            feature_list.append(p)

    data = pd.DataFrame(feature_list, index=person_list)
    data.to_csv('./DeepFeature/train6.csv', index= True)

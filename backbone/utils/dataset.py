'''
    Classes in this code are all for Road Extraction Task
'''

import os
import glob
import math
import cv2
import numpy as np
from torch.utils.data import Dataset
# from albumentations.pytorch import ToTensorV2

'''
    DeepglobeDataset : Normal Dataset for Deepglobe
    
    # input
    img_dir : dir of images (RGB image) 
    mask_dir : dir of mask  (1D binary image)
    
    # output
    image : [Batch, 3, H, W] / float / tensor
    mask : [Batch, H, W] / float / tensor (0 or 1 binary mask)
'''
class DeepglobeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.ids = self.get_ids_()
        
    def get_ids_(self):
        path_list = glob.glob(os.path.join(self.img_dir, '*'))
        ids = list(map(lambda x: os.path.basename(x).split('.')[0], path_list))
        return ids
    
    def __len__(self):
        return len(self.ids)
    
        
    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{id}.png')
        mask_path = os.path.join(self.mask_dir, f'{id}.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.where(mask >=1, 1, 0)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations['image'].float()
        mask = augmentations['mask'].float()
        
        data = {'image': image, 'mask': mask, 'image_path': img_path}
        return data

'''
    DeepglobeDatasetDobule : Multi-task for Deepglobe (2 tasks, only binary mask)
    
    # input
    img_dir : dir of images (RGB image) 
    mask_dir : dir of mask  (1D binary image)
    task_dir: dir of task mask (1D binary image) 
    
    # output
    image : [Batch, 3, H, W] / float / tensor
    segments : [mask, task] / list
        mask : [Batch, H, W] / float / tensor (0 or 1 binary mask for surface task)
        task : [Batch, H, W] / float / tensor (0 or 1 binary mask for specific task) 
'''
class DeepglobeDatasetDouble(Dataset):
    def __init__(self, img_dir, mask_dir, task_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.task_dir = task_dir
        self.transform = transform
    
        self.ids = self.get_ids_()
        
    def get_ids_(self):
        path_list = glob.glob(os.path.join(self.img_dir, '*'))
        ids = list(map(lambda x: os.path.basename(x).split('.')[0], path_list))
        return ids
    
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{id}.png')
        mask_path = os.path.join(self.mask_dir, f'{id}.png')
        task_path = os.path.join(self.task_dir, f'{id}.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.where(mask >=1, 1, 0) # 0 or 1
        task = cv2.imread(task_path)
        task = cv2.cvtColor(task, cv2.COLOR_BGR2RGB)
        task = np.where(task >=1, 1, 0)  # 0 or 1
        
        mask = mask[:,:,0] if mask.ndim ==3 else None # 3D mask to 1D [H, W, 3] -> [H, W]
        task = task[:,:,0] if task.ndim ==3 else None # 3D mask to 1D [H, W, 3] -> [H, W]

        augmentations = self.transform(image=image, masks=[mask, task])
        image = augmentations['image'].float()
        mask = augmentations['masks'][0].float()
        task = augmentations['masks'][1].float()
        segments = [mask, task]
        
        return image, segments
    
 
'''
    DeepglobeDatasetDobule : Multi-task for Deepglobe (2 tasks, only binary mask)
    
    # input
    img_dir : dir of images (RGB image) 
    mask_dir : dir of mask  (1D binary image)
    task_dir: dir of task mask (1D binary image) 
    
    # output
    image : [Batch, 3, H, W] / float / tensor
    segments : [mask, task] / list
        mask : [Batch, H, W] / float / tensor (0 or 1 binary mask for surface task)
        task : [Batch, H, W] / float / tensor (0 or 1 binary mask for specific task) 
'''    
class DeepglobeDatasetMulti(Dataset):
    '''
        task : surface, center, landcover
    '''
    def __init__(self, img_dir, mask_dir, center_dir, edge_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.center_dir = center_dir
        self.edge_dir = edge_dir
        self.transform = transform
    
        self.ids = self.get_ids_()
        
    def get_ids_(self):
        path_list = glob.glob(os.path.join(self.img_dir, '*'))
        ids = list(map(lambda x: os.path.basename(x).split('.')[0], path_list))
        return ids
    
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{id}.png')
        mask_path = os.path.join(self.mask_dir, f'{id}.png')
        center_path = os.path.join(self.center_dir, f'{id}.png')
        edge_path = os.path.join(self.edge_dir, f'{id}.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.where(mask >=1, 1, 0)
        center = cv2.imread(center_path)
        center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
        center = np.where(center >=1, 1, 0)
        edge = cv2.imread(edge_path)
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
        edge = np.where(edge >=1, 1, 0) 
        
        mask = mask[:,:,0] if mask.ndim ==3 else None
        center = center[:,:,0] if center.ndim ==3 else None
        edge = edge[:,:,0] if edge.ndim ==3 else None

        augmentations = self.transform(image=image, masks=[mask, center, edge])
        image = augmentations['image'].float()
        mask = augmentations['masks'][0].float()
        center = augmentations['masks'][1].float()
        edge = augmentations['masks'][2].float()
        segments = [mask, center, edge]
        
        return image, segments
    
    
class DeepglobeDatasetMulticlass(Dataset):
    '''
        task : surface, center, landcover
        Multiclass : edge -> landcover
    '''
    def __init__(self, img_dir, mask_dir, center_dir, landcover_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.center_dir = center_dir
        self.landcover_dir = landcover_dir
        self.transform = transform
    
        self.ids = self.get_ids_()
        
    def get_ids_(self):
        path_list = glob.glob(os.path.join(self.img_dir, '*'))
        ids = list(map(lambda x: os.path.basename(x).split('.')[0], path_list))
        return ids
    
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{id}.png')
        mask_path = os.path.join(self.mask_dir, f'{id}.png')
        center_path = os.path.join(self.center_dir, f'{id}.png')
        landcover_path = os.path.join(self.landcover_dir, f'{id}.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = np.where(mask >=1, 1, 0)
        center = cv2.imread(center_path)
        center = cv2.cvtColor(center, cv2.COLOR_BGR2RGB)
        center = np.where(center >=1, 1, 0)
        landcover = cv2.imread(landcover_path)
        landcover = cv2.cvtColor(landcover, cv2.COLOR_BGR2RGB) # Multi-class / index : 0 ~ 7
        # edge = np.where(edge >=1, 1, 0) 
        
        mask = mask[:,:,0] if mask.ndim ==3 else None
        center = center[:,:,0] if center.ndim ==3 else None
        landcover = landcover[:,:,0] if landcover.ndim ==3 else None

        augmentations = self.transform(image=image, masks=[mask, center, landcover])
        image = augmentations['image'].float()
        mask = augmentations['masks'][0].float()
        center = augmentations['masks'][1].float()
        landcover = augmentations['masks'][2].long() # Multi-class sholud be long dtype
        segments = [mask, center, landcover]
        
        return image, segments


class LoveDADataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.ids = self.get_ids_()
        
    def get_ids_(self):
        path_list = glob.glob(os.path.join(self.img_dir, '*'))
        ids = list(map(lambda x: os.path.basename(x).split('.')[0], path_list))
        return ids
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f'{id}.png')
        mask_path = os.path.join(self.mask_dir, f'{id}.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #     mask = np.where(mask >=1, 1, 0)
        if mask.ndim == 3:
            mask = mask[:,:,0]
        
        augmentations = self.transform(image=image, mask=mask)
        image = augmentations['image'].float()
        mask = augmentations['mask'].long()
        
        return image, mask

    

class PredictDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.paths = glob.glob(f'{self.img_dir}/*')
        self.ToTensor = ToTensorV2()

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.ToTensor(image=image)['image'].float()
        
        return image, path

def to_tiles(arr, tile_size=(1024,1024), overlap=0):
    if arr.ndim != 3:
        arr = np.expand_dims(arr, axis=2)
    h, w, channel = arr.shape # original image size
    t_h, t_w = tile_size # tile image size
    
    assert h >= t_h, f"source image size {(h, w)} should be bigger than tile size {tile_size}"
    assert w >= t_w, f"source image size {(h, w)} should be bigger than tile size {tile_size}"
    
    new_h = int(t_h - overlap) * math.ceil(h / (t_h - overlap)) + overlap 
    new_w = int(t_w - overlap) * math.ceil(w / (t_w - overlap)) + overlap 
    
    # padding
    pad_size_h = new_h - h
    pad_size_w = new_w - w
    up, bottom = math.floor(pad_size_h/2), math.ceil(pad_size_h/2)
    left, right = math.floor(pad_size_w/2), math.ceil(pad_size_w/2)
    arr_padded = np.pad(arr, ((up,bottom),(left,right), (0,0)), 'constant')
    pad_values = [up, bottom, left, right]
    
    # slice
    rows = int(math.floor(new_h / (t_h - overlap)))
    cols = int(math.floor(new_w / (t_w - overlap)))
    tiles = np.zeros((rows, cols, *tile_size, channel))
    for row in range(rows):
        for col in range(cols):
            start_h = (t_h - overlap) * row
            end_h   = start_h + t_h
            start_w = (t_w - overlap) * col
            end_w   = start_w + t_w
            tile = arr_padded[start_h:end_h, start_w:end_w]
    
            # 마지막 tile의 크기가 t_size보다 작은 경우
            # 즉, 마지막 tile의 끝 행렬이 arr_padded 범위를 벗어나는 경우
            if end_w > arr_padded.shape[1]:
                f_cols = cols - 1 # final cols
                continue
            elif end_h > arr_padded.shape[0]:
                f_rows = rows - 1 # final rows
                continue
            else:
                f_cols = cols
                f_rows = rows
                tiles[row,col,:] = tile        
    
    return tiles[:f_rows, :f_cols].astype(np.uint8), arr_padded.astype(np.uint8), pad_values
    
    
# LoveDA Example
# if __name__ == '__main__':
#     import albumentations as A
#     from torch.utils.data import DataLoader
#     from albumentations.pytorch import ToTensorV2
#     import numpy as np
#     transform_train = A.Compose([
#         A.Rotate(limit=40,p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         ToTensorV2()
#     ])
#     img_dir = '/home/yh.sakong/data/LoveDA/Train512/images'
#     mask_dir = '/home/yh.sakong/data/LoveDA/Train512/masks'
#     trainset = LoveDADataset(img_dir, mask_dir, transform=transform_train)
#     trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
#     img, mask = iter(trainloader).next()
#     print(len(trainloader.dataset))
#     print(len(trainloader))
#     print(img.shape)
#     print(mask.shape)


# Deepglobe Example
if __name__ == '__main__':
    import albumentations as A
    from torch.utils.data import DataLoader
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    transform_train = A.Compose([
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])
    
    # img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_image'
    # mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_segment'
    # center_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_centerline'
    # edge_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_edge'

    
    # img_dir = '/home/yh.sakong/data/massachusetts_road/png_500/train'
    # mask_dir = '/home/yh.sakong/data/massachusetts_road/png_500/train_labels'
    
    # trainset = DeepglobeDataset(img_dir, mask_dir, transform_train)
    # # trainset = DeepglobeDatasetMulti(img_dir, mask_dir, center_dir, edge_dir, transform_train)
    # trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    # print(len(trainset))
    
    # img, segments = iter(trainloader).next()
    # print(len(trainloader.dataset))
    # print(len(trainloader))
    # print('mask shape', segments.shape)
    # # print('center shape', segments[1].shape)
    # # print('edge shape', segments[2].shape)
    # print(len(trainloader.dataset))
    
    train_img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024/train_image'
    train_mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024/train_segment'
    train_building_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024_building/train'
    
    test_img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024/test_image'
    test_mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024/test_segment'
    test_building_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe1024_building/test'
    
    transform_train = A.Compose([
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()])
    
    transform_test = A.Compose([
        ToTensorV2()])

    trainset = DeepglobeDatasetDouble(train_img_dir, train_mask_dir, train_building_dir, transform_train)
    testset = DeepglobeDatasetDouble(test_img_dir, test_mask_dir, test_building_dir, transform_test)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    images, labels = iter(trainloader).next()
    
    print(images.shape)
    print(labels[0].shape)
    print(labels[1].shape)
    
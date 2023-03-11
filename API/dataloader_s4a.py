import gzip
import numpy as np
import os
import json
import random
import torch
import torch.utils.data as data

IMG_SIZE = 366

# Band names and their resolutions
BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}


SELECTED_CLASSES = [
    110,   # 'Wheat'
    120,   # 'Maize'
    140,   # 'Sorghum'
    150,   # 'Barley'
    160,   # 'Rye'
    170,   # 'Oats'
    330,   # 'Grapes'
    435,   # 'Rapeseed'
    438,   # 'Sunflower'
    510,   # 'Potatoes'
    770,   # 'Peas'
]

LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
LINEAR_ENCODER[0] = 0

NORMALIZATION_DIV = 10000


def min_max_normalize(image, percentile=2):
    image = image.astype('float32')

    percent_min = np.percentile(image, percentile, axis=(0, 1))
    percent_max = np.percentile(image, 100-percentile, axis=(0, 1))

    mask = np.mean(image, axis=2) != 0
    if image.shape[1] * image.shape[0] - np.sum(mask) > 0:
        mdata = np.ma.masked_equal(image, 0, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image-percent_min) / (percent_max - percent_min)
    norm[norm < 0] = 0
    norm[norm > 1] = 1
    norm = norm * mask[:, :, np.newaxis]
    # norm = (norm * 255).astype('uint8') * mask[:,:,np.newaxis]

    return norm


class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, ann):
        image, landmarks = img, ann

        _, _, h, w = image.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, :, top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks[top: top + new_h, left: left + new_w]
        img = image
        ann = landmarks

        return img, ann


class NpyPADDataset(data.Dataset):
    '''
        ├── dataset
        │   ├── my_dataset
        │   │   ├── scenario1_filename.json
        │   │   ├── scenario2_filename.json
        │   │   ├── ...
        │   │   ├── nrgb (B02 B03 B04 B08)
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── rdeg (B05, B06, B07, B8A, B11, B12)
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── label
        │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   ├── zzz{seg_map_suffix}
    '''

    def __init__(
            self,
            root_dir: str = None,
            img_dir: str = None,
            ann_dir: str = None,
            band_mode: str = 'nrgb',
            bands: list = None,
            linear_encoder: dict = None,
            start_month: int = 0,
            end_month: int = 12,
            output_size: tuple = (64, 64),
            binary_labels: bool = False,
            return_parcels: bool = False,
            mode: str = 'test',
            scenario: int = 1,
            get_ann: bool = False,
    ) -> None:
        '''
        Args:
            bands: list of str, default None
                A list of the bands to use. If None, then all available bands are
                taken into consideration. Note that the bands are given in a two-digit
                format, e.g. '01', '02', '8A', etc.
            linear_encoder: dict, default None
                Maps arbitrary crop_ids to range 0-len(unique(crop_id)).
            output_size: tuple of int, default None
                If a tuple (H, W) is given, then the output images will be divided
                into non-overlapping subpatches of size (H, W). Otherwise, the images
                will retain their original size.
            binary_labels: bool, default False
                Map categories to 0 background, 1 parcel.
            mode: str, ['train', 'val', 'test']
                The running mode. Used to determine the correct path for the median files.
            return_parcels: boolean, default False
                If True, then a boolean mask for the parcels is also returned.
        '''

        self.band_mode = band_mode
        if band_mode == 'nrgb':
            self.bands = ['B02', 'B03', 'B04', 'B08']

        elif band_mode == 'rdeg':
            self.bands = ['B02', 'B03', 'B04', 'B08',
                          'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

        else:
            raise RuntimeError

        self.return_parcels = return_parcels
        self.binary_labels = binary_labels

        if output_size is None:
            output_size = [IMG_SIZE, IMG_SIZE]
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int),\
            'sub-patches dims must be integers!'
        assert output_size[0] == output_size[1], \
            f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'
        self.output_size = [int(dim) for dim in output_size]

        # We index based on year, number of bins should be the same for every year
        # therefore, calculate them using a random year

        self.img_dir = img_dir if img_dir else os.path.join(root_dir, 'nrgb')
        self.ann_dir = ann_dir if ann_dir else os.path.join(root_dir, 'label')
        self.root_dir = root_dir if root_dir else os.path.dirname(self.img_dir)

        self.start_month = start_month - 1
        self.end_month = end_month - 1
        self.linear_encoder = LINEAR_ENCODER
        self.min_max_normalize = True
        self.get_ann = get_ann

        self.mode = mode
        self.scenario = scenario
        assert self.mode in ['train', 'val', 'test'], \
            "variable mode should be 'train' or 'val' or 'test'"
        assert self.scenario == 1 or self.scenario == 2, \
            'variable scenario should be 1 or 2'

        with open(os.path.join(self.root_dir, f"scenario{self.scenario}_filename.json"), "r") as st_json:
            self.img_infos = json.load(st_json)[mode]
            self.img_infos = [f + '.npy' for f in self.img_infos]

        if output_size[0] != IMG_SIZE:
            self.transforms = RandomCrop(output_size[0])
        else:
            self.transforms = None

        # copied from dataloader_moving_mnist
        self.mean = 0
        self.std = 1

        print('Rootdir: {}'.format(self.root_dir))
        print('Scenario: {}, MODE: {}, length of datasets: {}'.format(
            self.scenario, self.mode, len(self.img_infos)))
        print('Acquired Data Month: From {} to {}'.format(
            self.start_month + 1, self.end_month + 1))
        print(
            f'Data shape: [T, C, H, W] ({self.end_month - self.start_month}, {len(self.bands)}, {output_size[0]}, {output_size[0]})')

    def prepare_train_img(self, idx: int) -> dict:
        if self.band_mode == 'nrgb':
            readpath = os.path.join(self.img_dir, self.img_infos[idx])
            img = np.load(readpath)
        elif self.band_mode == 'rdeg':
            readpath = os.path.join(self.img_dir, self.img_infos[idx])
            img = np.load(readpath)

            rdegpath = os.path.join(os.path.dirname(
                self.img_dir), 'rdeg', self.img_infos[idx])
            rdeg = np.load(rdegpath)
            img = np.stack([img, rdeg], axis=1)
        else:
            raise RuntimeError

        annpath = os.path.join(self.ann_dir, self.img_infos[idx])
        ann = np.load(annpath)

        img = img[self.start_month:self.end_month]

        if self.transforms:
            img, ann = self.transforms(img, ann)

        return img, ann

    def _normalize(self, img):
        if self.min_max_normalize:
            T, C, H, W = img.shape
            img = img.reshape(T*C, H, W)
            img = min_max_normalize(img.transpose(1, 2, 0), percentile=0.5)
            img = img.transpose(2, 0, 1)
            img = img.reshape(T, C, H, W)
        else:
            img = np.divide(img, NORMALIZATION_DIV)  # / 10000
        return img

    def __getitem__(self, idx: int) -> dict:
        img, ann = self.prepare_train_img(idx)

        # Normalize data to range [0-1]
        img = self._normalize(img)

    ########TODO########
        out = {}
        if self.return_parcels:
            parcels = ann != 0
            out['parcels'] = parcels

        if self.binary_labels:
            # Map 0: background class, 1: parcel
            ann[ann != 0] = 1
        else:
            # Map labels to 0-len(unique(crop_id)) see config
            # labels = np.vectorize(self.linear_encoder.get)(labels)
            _ = np.zeros_like(ann)
            for crop_id, linear_id in self.linear_encoder.items():
                _[ann == crop_id] = linear_id
            ann = _

        # # # Map all classes NOT in linear encoder's values to 0
        # ann[~np.isin(ann, list(self.linear_encoder.values()))] = 0

        input = img[:8]
        output = img[8:]

        output = torch.from_numpy(output).contiguous().float()
        input = torch.from_numpy(input).contiguous().float()

        if self.get_ann:
            ann = ann.astype('float32')
            _ = np.zeros_like(ann)
            for crop_id, linear_id in LINEAR_ENCODER.items():
                _[ann == crop_id] = linear_id
            ann = _ / len(SELECTED_CLASSES)
            ann = torch.from_numpy(ann).contiguous().long()
            return input, output, ann
        else:
            return input, output

    def __len__(self):
        return len(self.img_infos)


def load_data(batch_size, val_batch_size, num_workers, data_root):
    # get_ann 옵션 추가
    train_set = NpyPADDataset(
        root_dir=data_root, band_mode='nrgb', start_month=1, end_month=13, get_ann=False, mode='train')
    test_set = NpyPADDataset(
        root_dir=data_root, band_mode='nrgb', start_month=1, end_month=13, get_ann=False, mode='val')

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, num_workers=num_workers)

    return dataloader_train, dataloader_validation, dataloader_test


if __name__ == '__main__':
    rootdir = '/home/jovyan/shared_volume/data/newdata'
    dataset = NpyPADDataset(
        root_dir=rootdir, band_mode='nrgb', start_month=1, end_month=13, get_ann=True, mode='val')
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
    
    # print(dataset[0][2])

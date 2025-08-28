from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, mode='storm', flag='train'):
        self.mode=  mode

        if mode == 'storm':
            dataset = np.load('storm_dataset4.npz', allow_pickle=True, mmap_mode='r')
        elif mode == 'nonstorm':
            dataset = np.load('nonstorm_dataset4.npz', allow_pickle=True, mmap_mode='r')
        else:
            raise NotImplementedError

        self.x = dataset['inp']
        self.y = dataset['tar']
        self.inp_t = dataset['inp_time']
        self.tar_t = dataset['tar_time']

        year_start = self.inp_t[:, 0].astype('datetime64[Y]').astype(int) + 1970
        year_end = self.inp_t[:, -1].astype('datetime64[Y]').astype(int) + 1970

        x, y = self._data_preprocess()

        if flag == 'train':
            mask = (year_start >= 2000) & (year_start <= 2014) & (year_end >= 2000) & (year_end <= 2014)
            train_idx = np.nonzero(mask)[0]

            if mode == 'nonstorm':
                random_index = np.random.randint(0, len(train_idx), 34044 * 3)
                self.data_x = x[random_index]
                self.data_y = y[random_index]

            elif mode == 'storm':
                self.data_x = x[train_idx]
                self.data_y = y[train_idx]
            else:
                print("mode must be 'storm' or 'nonstorm'")
                raise NotImplementedError

        elif flag == 'test':
            mask = (year_start >= 2015) & (year_start <= 2024) & (year_end >= 2015) & (year_end <= 2024)
            test_idx = np.nonzero(mask)[0]
            self.data_x = x[test_idx]
            self.data_y = y[test_idx]
        else:
            print("flag must be 'train' or 'test'")
            raise NotImplementedError

        print(self.data_x.shape, self.data_y.shape)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data_x[idx])
        y = torch.Tensor(self.data_y[idx])
        return x, y

    def _data_preprocess(self):
        # --- solar wind parameters --- #
        def interpolate_time_axis(data, sentinel=-1):
            N, T, F = data.shape
            idx = np.arange(T)
            out = data.astype('float32').copy()

            for n in range(N):
                for f in range(F):
                    y = out[n, :, f]
                    nan_mask = np.isnan(y)
                    if nan_mask.all():
                        # 전부 NaN인 경우 원하는 값으로 채우기
                        out[n, :, f] = sentinel
                    else:
                        valid_x = idx[~nan_mask]
                        valid_y = y[~nan_mask]
                        # 양끝 NaN은 양 끝값으로 익스텐션
                        out[n, nan_mask, f] = np.interp(
                            idx[nan_mask], valid_x, valid_y,
                            left=valid_y[0], right=valid_y[-1]
                        )
            return out

        x = interpolate_time_axis(self.x)

        # --- target --- #
        tar_ = self.y.astype('float32')
        tar_ = pd.DataFrame(tar_).interpolate().values

        x[:, :, 6:10] = np.abs(x[:, :, 6:10])   # convert solar wind speed to positive value
        y = tar_

        mins = x.min(axis=(0, 1), keepdims=True)
        maxs = x.max(axis=(0, 1), keepdims=True)

        x_norm = (x - mins) / (maxs - mins)
        x_norm[:, :, -1] = x[:, :, -1] / 12.

        y_norm = y / 12.

        if self.mode == 'storm':
            idx = np.delete(np.arange(19), np.arange(10, 14))
            return x_norm[..., idx], y_norm
        else:
            return x_norm, y_norm

if __name__ == '__main__':
    train_dataset = CustomDataset(mode='storm', flag='train')
    print(len(train_dataset))

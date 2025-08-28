import numpy as np

class Dataloader:
    def __init__(self, mode, n_features=6):
        super(Dataloader, self).__init__()
        self.n_features = n_features
        dataset = np.load('dataset.npz', allow_pickle=True)
        self.inp, self.tar = self._data_preprocessing(dataset['inp'], dataset['tar'])
        self.mode = mode

    def get_dataset(self):
        test_index = np.concatenate([np.arange(1124, 1130), np.arange(4101, 4128), np.arange(5010, 5035)])
        mask = ~np.isin(np.arange(len(self.inp)), test_index)
        train_index = np.arange(len(self.inp))[mask]

        if self.mode == 'train':
             return self.inp[train_index], self.tar[train_index]

        elif self.mode == 'test':
             return self.inp[test_index], self.tar[test_index]

        else:
            raise ValueError('Invalid mode')

    def interpolate_time_axis(self, data):
        N, T, F = data.shape
        idx = np.arange(T)
        out = data.astype('float32').copy()

        for n in range(N):
            for f in range(F):
                y = out[n, :, f]
                nan_mask = np.isnan(y)

                valid_x = idx[~nan_mask]
                valid_y = y[~nan_mask]
                # 양끝 NaN은 양 끝값으로 익스텐션
                out[n, nan_mask, f] = np.interp(
                    idx[nan_mask], valid_x, valid_y,
                    left=valid_y[0], right=valid_y[-1]
                )
        return out

    def _data_preprocessing(self, x, y):
        interp_inp = self.interpolate_time_axis(x)
        interp_tar = self.interpolate_time_axis(y[..., np.newaxis])

        min_values = np.min(interp_inp, axis=(0, 1))
        max_values = np.max(interp_inp, axis=(0, 1))

        interp_inp_norm = np.full(interp_inp.shape, np.nan)
        for i in range(self.n_features):
            interp_inp_norm[:, :, i] = 2 * (interp_inp[:, :, i] - min_values[i]) / (max_values[i] - min_values[i]) - 1

        interp_tar_norm = 2 * (interp_tar - min_values[1]) / (max_values[1] - min_values[1]) - 1
        interp_tar_norm = np.squeeze(interp_tar_norm, axis=-1)
        return interp_inp_norm, interp_tar_norm

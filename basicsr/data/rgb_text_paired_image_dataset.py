import os.path as osp

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, img2tensor, imfrombytes, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


def _b2_root_list(value, option_name):
    roots = value if isinstance(value, list) else [value]
    if not roots or any(not isinstance(root, str) or not root for root in roots):
        raise ValueError(f'{option_name} must contain one or more non-empty paths.')
    return roots


@DATASET_REGISTRY.register()
class RGBTextPairedImageDataset(data.Dataset):
    """Stage-1 B2 dataset: paired LR RGB, HR RGB, and one global caption.

    B2 change note: the image pipeline is identical to PairedImageDataset.
    The only added input is a UTF-8 caption paired by the GT basename. The
    caption describes the full image and therefore is intentionally unchanged
    by the synchronized LR/GT crop and augmentation.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = dict(opt['io_backend'])
        if self.io_backend_opt.get('type') != 'disk':
            raise NotImplementedError('RGBTextPairedImageDataset currently supports disk folders only.')

        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.filename_tmpl_text = opt.get('filename_tmpl_text', '{}')
        self.text_file_ext = str(opt.get('text_file_ext', '.txt'))
        if not self.text_file_ext.startswith('.'):
            self.text_file_ext = f'.{self.text_file_ext}'
        self.text_encoding = opt.get('text_encoding', 'utf-8')

        gt_roots = _b2_root_list(opt['dataroot_gt'], 'dataroot_gt')
        lq_roots = _b2_root_list(opt['dataroot_lq'], 'dataroot_lq')
        text_roots = _b2_root_list(opt['dataroot_text'], 'dataroot_text')
        if not (len(gt_roots) == len(lq_roots) == len(text_roots)):
            raise ValueError('dataroot_gt, dataroot_lq, and dataroot_text must have equal lengths.')

        self.paths = []
        for gt_root, lq_root, text_root in zip(gt_roots, lq_roots, text_roots):
            lq_names = set(scandir(lq_root))
            text_names = set(scandir(text_root))
            for gt_name in sorted(scandir(gt_root)):
                basename, image_ext = osp.splitext(osp.basename(gt_name))
                lq_name = f'{self.filename_tmpl.format(basename)}{image_ext}'
                text_name = f'{self.filename_tmpl_text.format(basename)}{self.text_file_ext}'
                if lq_name not in lq_names:
                    raise FileNotFoundError(f'Missing paired LR RGB image: {osp.join(lq_root, lq_name)}')
                if text_name not in text_names:
                    raise FileNotFoundError(f'Missing paired text caption: {osp.join(text_root, text_name)}')
                self.paths.append({
                    'gt_path': osp.join(gt_root, gt_name),
                    'lq_path': osp.join(lq_root, lq_name),
                    'text_path': osp.join(text_root, text_name),
                })

        if not self.paths:
            raise ValueError('RGBTextPairedImageDataset found no paired samples.')

    def __getitem__(self, index):
        if self.file_client is None:
            backend_type = self.io_backend_opt.pop('type')
            self.file_client = FileClient(backend_type, **self.io_backend_opt)

        paths = self.paths[index]
        img_gt = imfrombytes(self.file_client.get(paths['gt_path'], 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(paths['lq_path'], 'lq'), float32=True)
        with open(paths['text_path'], 'r', encoding=self.text_encoding) as text_file:
            text = text_file.read().strip()
        if not text:
            raise ValueError(f'B2 requires a non-empty text caption: {paths["text_path"]}')

        scale = self.opt['scale']
        if self.opt['phase'] == 'train':
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.opt['gt_size'], scale, paths['gt_path'])
            img_gt, img_lq = augment(
                [img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if self.opt.get('color') == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[:img_lq.shape[0] * scale, :img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'text': text,
            'gt': img_gt,
            'lq_path': paths['lq_path'],
            'text_path': paths['text_path'],
            'gt_path': paths['gt_path'],
        }

    def __len__(self):
        return len(self.paths)

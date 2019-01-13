import os
import logging
from mxboard import SummaryWriter
import glob
import itertools
import numpy as np
import nibabel as nib

import mxnet as mx
from mxnet import gluon, autograd, ndarray as nd
from mxnet.gluon.nn import Activation, Conv3D, Conv3DTranspose, \
    BatchNorm, HybridSequential, HybridBlock, Dropout, MaxPool3D

import gluoncv
from gluoncv.data.segbase import SegmentationDataset

from scipy.ndimage import affine_transform
from math import pi
from transforms3d import affines, euler

################################################################################
# MRI segmentation dataset

class MRISegDataset(SegmentationDataset):
    """Semantic segmentation directory dataset class for MRI volumetric data"""
    NUM_CLASS = 4
    def __init__(self, root, split='train', fold_inds=None, mode=None, 
                 transform=None, means=None, stds=None, lesion_frac=0.9, warp_params=None, **kwargs):
        super(MRISegDataset, self).__init__(root, split, mode, transform, warp_params, **kwargs)
        self.fold_inds = fold_inds
        self.means = means
        self.stds = stds
        self.lesion_frac = lesion_frac
        self.warp_params = warp_params

        # Get input file path lists
        if split == 'train':
            #import pdb; pdb.set_trace()
            self._dataset_root = os.path.join(root, 'training')
            self.sub_dirs = glob.glob(self._dataset_root + '/*/*/')
            if fold_inds is not None:
                mask = np.ones(len(self.sub_dirs), np.bool)
                mask[fold_inds] = 0
                ikeep = np.arange(0, len(self.sub_dirs))[mask]
                self.sub_dirs = np.array(self.sub_dirs)[ikeep]
                if self.means is not None: self.means = self.means[ikeep]
                if self.stds  is not None: self.stds  = self.stds[ikeep]
        elif split == 'val':
            self._dataset_root = os.path.join(root, 'training')
            self.sub_dirs = glob.glob(self._dataset_root + '/*/*/')
            if fold_inds is not None:
                fold_dirs = np.array(self.sub_dirs)[fold_inds]
                self.sub_dirs = fold_dirs
                if self.means is not None: self.means = self.means[fold_inds]
                if self.stds  is not None: self.stds  = self.stds[fold_inds]
        elif split == 'test':
            self._dataset_root = os.path.join(root, 'validation')
            self.sub_dirs = glob.glob(self._dataset_root + '/*/')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        _sub_name = os.path.basename(os.path.dirname(self.sub_dirs[idx]))

        # Load multichannel input data
        channels = ['flair', 't1', 't1ce', 't2']
        img_paths = [os.path.join(self.sub_dirs[idx], _sub_name + '_' + channel + '.nii.gz') for channel in channels]
        img = []
        for img_path in img_paths:
            img.append(nib.load(img_path).get_fdata())
        img = np.array(img)
        img = np.flip(img, 2) # Correct AP orientation
        
        # Load segmentation label map
        target = None
        if self.split is not 'test':
            target = nib.load(os.path.join(self.sub_dirs[idx], _sub_name + '_seg.nii.gz')).get_fdata()
            target[target==4] = 3 # Need to have consecutive integers [0, n_classes) for training
        else:
            target = np.zeros_like(img[0,:]) # dummy segmentation
        target = np.expand_dims(target, axis=0)
        target = np.flip(target, 2) # Correct AP orientation

        # Data augmentation
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))

        # Routine img specific processing (normalize, etc.)
        if self.transform is not None:
            img = self.transform(img, self.means[idx], self.stds[idx])

        return img, target

    def _sync_transform(self, img, mask):
        crop_size = self.crop_size
        
        # Random LR flip
        if np.random.random() < 0.5:
            img  = np.fliplr(img)
            mask = np.fliplr(mask)

        # Pad if smaller than crop_size
        if any(np.array(img.shape[1:]) < crop_size):
            img, mask = img_pad(img, mask, crop_size)
            
        # Random crop if larger than crop_size
        if any(np.array(img.shape[1:]) > crop_size):
            img, mask = img_crop(img, mask, crop_size, self.lesion_frac)

        # Random affine
        if self.warp_params:
            img, mask = img_warp(img, mask, self.warp_params['theta_max'], 
                                            self.warp_params['offset_max'],
                                            self.warp_params['scale_max'],
                                            self.warp_params['shear_max'])
        
        # final transform to mxnet NDArray
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _val_sync_transform(self, img, mask):
        crop_size = self.crop_size

        # Pad if smaller than crop_size
        if any(np.array(img.shape[1:]) < crop_size):
            img, mask = img_pad(img, mask, crop_size)

        # final transform to mxnet NDArray
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.sub_dirs)

    def paths(self):
        return self.sub_dirs

    @property
    def classes(self):
        """Category names."""
        return ('background', 'necrotic', 'edema', 'enhancing')

class MRISegDataset4D(MRISegDataset):
    """Dataset class with all inputs and GT seg combined into a single 4D NifTI"""
    def __getitem__(self, idx):
        #import pdb; pdb.set_trace()
        _sub_name = os.path.basename(os.path.dirname(self.sub_dirs[idx]))

        # Load multichannel input data
        img_path = os.path.join(self.sub_dirs[idx], _sub_name + '_' + '4D' + '.nii.gz')
        img_raw = nib.load(img_path).get_fdata()
        img_raw = img_raw.transpose((3,0,1,2))
        img_raw = np.flip(img_raw, 2) # Correct AP orientation
        img = img_raw[0:4]
        
        # Load segmentation label map
        if self.split is not 'test':
            target = img_raw[4]
            target[target==4] = 3 # Need to have consecutive integers [0, n_classes) for training
        else:
            target = np.zeros_like(img[0,:]) # dummy segmentation
        target = np.expand_dims(target, axis=0)

        # Data augmentation
        if self.mode == 'train':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))

        # Routine img specific processing (normalize, etc.)
        if self.transform is not None:
            img = self.transform(img, self.means[idx], self.stds[idx])

        return img, target

def img_pad(img, mask, pad_dims):
    """Pad input image vol to given voxel dims"""
    x_pad0, x_pad1, y_pad0, y_pad1, z_pad0, z_pad1 = 0,0,0,0,0,0
    dims = img.shape[1:]
    if dims[0] < pad_dims[0]:
        x_pad0 = (pad_dims[0] - dims[0]) // 2
        x_pad1 = pad_dims[0] - dims[0] - x_pad0
    if dims[1] < pad_dims[1]:
        y_pad0 = (pad_dims[1] - dims[1]) // 2
        y_pad1 = pad_dims[1] - dims[1] - y_pad0
    if dims[2] < pad_dims[2]:
        z_pad0 = (pad_dims[2] - dims[2]) // 2
        z_pad1 = pad_dims[2] - dims[2] - z_pad0

    padding = ((0, 0), (x_pad0, x_pad1), (y_pad0, y_pad1), (z_pad0, z_pad1))

    img  = np.pad(img,  padding, 'constant', constant_values=0)
    mask = np.pad(mask, padding, 'constant', constant_values=0)
    return img, mask

def img_unpad(img, dims):
    """Unpad image vol back to original input dimensions"""
    pad_dims = img.shape
    xmin, ymin, zmin = 0, 0, 0
    if pad_dims[0] > dims[0]:
        xmin = (pad_dims[0] - dims[0]) // 2
    if pad_dims[1] > dims[1]:
        ymin = (pad_dims[1] - dims[1]) // 2
    if pad_dims[2] > dims[2]:
        zmin = (pad_dims[2] - dims[2]) // 2
    return img[xmin : xmin + dims[0],
               ymin : ymin + dims[1],
               zmin : zmin + dims[2]]

def img_crop(img, mask, crop_size, lesion_frac=0.9):
    """Sample a random image subvol/patch from larger input vol"""
    # Pick the location for the patch centerpoint
    if np.random.random() < lesion_frac:
        good_inds = (mask.squeeze() != 0).nonzero() # sample all lesion voxels
    else:
        good_inds = (img[0,] != 0).nonzero() # sample all brain voxels
    i_center = np.random.randint(len(good_inds[0]))
    xmin = good_inds[0][i_center] - crop_size[0] // 2
    ymin = good_inds[1][i_center] - crop_size[1] // 2
    zmin = good_inds[2][i_center] - crop_size[2] // 2

    # Make sure centerpoint is not too small
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if zmin < 0: zmin = 0

    # Make sure centerpoint is not too big
    max_sizes = np.array(img.shape[1:]) - crop_size
    if xmin > max_sizes[0]: xmin = max_sizes[0]
    if ymin > max_sizes[1]: ymin = max_sizes[1]
    if zmin > max_sizes[2]: zmin = max_sizes[2]
    
    img  =  img[:, xmin : xmin + crop_size[0], 
                   ymin : ymin + crop_size[1],
                   zmin : zmin + crop_size[2]]
    mask = mask[:, xmin : xmin + crop_size[0], 
                   ymin : ymin + crop_size[1],
                   zmin : zmin + crop_size[2]]
    return img, mask

def img_warp(img, mask, theta_max=15, offset_max=0, scale_max=1.1, shear_max=0.1):
    """Training data augmentation with random affine transformation"""
    # Rotation
    vec = np.random.normal(0, 1, 3)
    vec /= np.sqrt(np.sum(vec ** 2))
    theta = np.random.uniform(- theta_max, theta_max, 1) * pi / 180
    R = euler.axangle2mat(vec, theta)
    
    # Scale/zoom
    sign = -1 if np.random.random() < 0.5 else 1
    Z = np.ones(3) * np.random.uniform(1, scale_max, 1) ** sign
    
    # Translation
    c_in = np.array(img.shape[1:]) // 2
    offset = np.random.uniform(- offset_max, offset_max, 3)
    T = - (c_in).dot((R * Z).T) + c_in + offset
    
    # Shear
    S = np.random.uniform(- shear_max, shear_max, 3)
    
    # Compose affine
    mat = affines.compose(T, R, Z, S)
    
    # Apply warp
    img_warped  = np.zeros_like(img)
    mask_warped = np.zeros_like(mask)
    for i in range(len(img)):
        img_warped[i,] = affine_transform(img[i,], mat, order=1) # Trilinear
    mask_warped[0,] = affine_transform(mask[0,], mat, order=0)   # Nearest neighbor
    
    return img_warped, mask_warped

def brats_transform(img, means, stds):
    """Routine image-specific processing (e.g. normalization)"""
    means = means.reshape(-1,1,1,1)
    stds  = stds.reshape(-1,1,1,1)
    return (img - means) / stds

################################################################################
# U-Net

def conv_block(channels, num_convs=2, use_bias=False, use_global_stats=False, **kwargs):
    """Define U-Net convolution block"""
    out = HybridSequential(prefix="")
    with out.name_scope():
        for _ in range(num_convs):
            out.add(Conv3D(channels=channels, kernel_size=3, padding=1, use_bias=use_bias))
            out.add(Activation('relu'))
            out.add(BatchNorm(use_global_stats=use_global_stats)) #BN after relu seems to be the more recommended option. 
    return out

class UnetSkipUnit(HybridBlock):
    """Define U-Net skip block"""
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False, use_dropout=False, use_bias=False, **kwargs):
        super(UnetSkipUnit, self).__init__()
        
        with self.name_scope():
            self.outermost = outermost
            
            downsample = MaxPool3D(pool_size=2, strides=2)
            upsample = Conv3DTranspose(channels=outer_channels, kernel_size=2, padding=0, strides=2, use_bias=use_bias)
            head = Conv3D(channels=outer_channels, kernel_size=1)
            
            self.model = HybridSequential()
            if not outermost:
                self.model.add(downsample)
            self.model.add(conv_block(inner_channels, use_bias=use_bias, **kwargs))
            if not innermost:
                self.model.add(inner_block)
                self.model.add(conv_block(inner_channels, use_bias=use_bias, **kwargs))
            if not outermost:
                self.model.add(upsample)
            if outermost:
                if use_dropout:
                    self.model.add(Dropout(rate=0.1))
                self.model.add(head)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)

class UnetGenerator(HybridBlock):
    """Define recursive U-Net generator"""
    def __init__(self, num_downs=4, classes=2, ngf=64, **kwargs):
        super(UnetGenerator, self).__init__()

        #Recursively build Unet from the inside out
        unet = UnetSkipUnit(ngf * 2 ** num_downs, ngf * 2 ** (num_downs-1), innermost=True, **kwargs)
        for depth in range(num_downs-1, 0, -1):
            unet = UnetSkipUnit(ngf * 2 ** depth, ngf * 2 ** (depth-1), unet, **kwargs)
        unet = UnetSkipUnit(ngf, classes, unet, outermost=True, **kwargs)
        
        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

################################################################################
# Inference

# ET = enhancing tumor (blue)                                           = label 4
# TC = tumor core             = ET + non-enhancing tumor/necrosis (red) = label 4 + 1
# WT = whole tumor            = TC + edema (green)                      = label 4 + 1 + 2

def brats_validate(model, data_loader, crop_size, overlap, ctx):
    """Predict segs from val data, compare to ground truth val segs, and calculate val dice metrics"""
    # Setup metric dictionary
    metrics = init_brats_metrics()

    # Get patch index iterator
    dims = data_loader._dataset[0][1].shape[1:]
    patch_iter = get_patch_iter(dims, crop_size, overlap)
    
    # Iterate over validation subjects
    for i, (data, label) in enumerate(data_loader):  

        # Iterate over patches
        for inds in patch_iter:
            # Extract patch
            data_patch  = get_patch(data, inds).as_in_context(ctx)
            label_patch = get_patch(label, inds)
            label_mask = label_patch.squeeze().asnumpy()
            
            # Run patch through net
            output_mask = get_output_mask(model, data_patch).asnumpy()

            # Update metrics
            for _, metric in metrics.items():
                label_mask_bin  = np.isin(label_mask, metric['labels'])
                output_mask_bin = np.isin(output_mask, metric['labels'])
                metric['tp']  += np.sum(label_mask_bin * output_mask_bin)
                metric['tot'] += np.sum(label_mask_bin) + np.sum(output_mask_bin)

    # Calculate overall metrics
    for _, metric in metrics.items():
            metric['DSC'] = 2 * metric['tp'] / metric['tot']
    return metrics

def brats_predict(model, data, crop_size, overlap, n_classes, ctx): 
    """Apply model to predict seg of unknown/test data"""
    # Get patch index iterator
    dims = data.squeeze().shape[1:]
    patch_iter = get_patch_iter(dims, crop_size, overlap)
    
    # Initialize output vol
    if overlap != 0:
        mask = - nd.ones(dims + (len(patch_iter), ), ctx=mx.cpu())
    else:
        mask = nd.zeros(dims, ctx=mx.cpu())
    
    # Iterate over patches 
    for i, inds in enumerate(patch_iter):
        data_patch = get_patch(data, inds).as_in_context(ctx)
        output_mask = get_output_mask(model, data_patch)
        mask = put_patch(mask, output_mask.as_in_context(mx.cpu()), inds, i)
    
    mask = mask.asnumpy()
    
    # If overlapping patches, get class prediction by majority vote (i.e. mode)
    if mask.shape != dims:
        mask = mode(mask, n_classes=n_classes)
        mask = mask.squeeze()
    return mask

def get_patch_iter(dims, crop_size, overlap):
    """Wrapper to get patch iterator"""
    x_patches = get_patch_inds(dims[0], crop_size[0], overlap) 
    y_patches = get_patch_inds(dims[1], crop_size[1], overlap)
    z_patches = get_patch_inds(dims[2], crop_size[2], overlap)
    patch_iter = list(itertools.product(x_patches, y_patches, z_patches)) 
    return patch_iter 

def get_patch_inds(axis_dim, crop_size, overlap):
    """Get list of indices needed to tile patches across entire input vol"""
    if crop_size > axis_dim:
        i_start = [0]
        i_end = [axis_dim]
    else:
        n_overlap = int(np.floor(crop_size * overlap))
        i_start = np.arange(0, axis_dim - n_overlap, crop_size - n_overlap)
        i_end = i_start + crop_size
        # Scoot the last patch back so it's not hanging off the edge
        if i_end[-1] > axis_dim:
            i_start[-1] = axis_dim - crop_size
            i_end[-1]   = axis_dim 
    return [[x, y] for x, y in zip(i_start, i_end)]

def get_patch(data, inds):
    """Extract patch data ndarray from a larger image vol"""
    x_inds = inds[0]
    y_inds = inds[1]
    z_inds = inds[2]
    data_patch  =  data[:, :, x_inds[0] : x_inds[1],
                              y_inds[0] : y_inds[1],
                              z_inds[0] : z_inds[1]]
    return data_patch

def put_patch(data, data_patch, inds, i):
    """Place patch data ndarray back into a larger image vol"""
    x_inds = inds[0]
    y_inds = inds[1]
    z_inds = inds[2]
    if np.ndim(data_patch) == np.ndim(data):
        data[x_inds[0] : x_inds[1],
             y_inds[0] : y_inds[1],
             z_inds[0] : z_inds[1]] = data_patch
    else:
        data[x_inds[0] : x_inds[1],
             y_inds[0] : y_inds[1],
             z_inds[0] : z_inds[1], i] = data_patch
    return data

def get_output_mask(model, data):
    """Wrapper for model prediction"""
    output = model(data)
    output_mask = output.argmax_channel().squeeze()
    return output_mask

################################################################################
# Misc helper functions

def init_brats_metrics():
    """Initialize dict for BraTS Dice metrics"""
    metrics = {}
    metrics['ET'] = {'labels': [3]}
    metrics['TC'] = {'labels': [1, 3]}
    metrics['WT'] = {'labels': [1, 2, 3]}
    for _, value in metrics.items():
        value.update({'tp':0, 'tot':0})
    return metrics

def calc_brats_metrics(label_mask, output_mask):
    """Calculate BraTS Dice metrics (ET, TC, WT)"""
    metrics = init_brats_metrics()
    for _, metric in metrics.items():
        label_mask_bin  = np.isin(label_mask, metric['labels'])
        output_mask_bin = np.isin(output_mask, metric['labels'])
        metric['tp']  = np.sum(label_mask_bin * output_mask_bin)
        metric['tot'] = np.sum(label_mask_bin) + np.sum(output_mask_bin)
        metric['DSC'] = 2 * metric['tp'] / metric['tot']
    return metrics

def dsc(truth, pred):
    """Dice Sorenson (similarity) Coefficient
    (For the simple binary or overall-multiclass case)
    """
    tp = truth == pred
    tp = tp * (truth != 0)
    return 2 * np.sum(tp) / (np.sum(truth != 0) + np.sum(pred != 0))

def get_k_folds(n, k, seed=None):
    """Simple cross-validation index generator"""
    np.random.seed(seed)
    x = np.arange(n)
    np.random.shuffle(x)
    np.random.seed()
    return [x[i::k] for i in range(k)]

def mode(x, n_classes):
    """Calculate the mode (i.e. ensemble vote) over a set of overlapping prediction patches"""
    dims = x.shape[:-1] + (n_classes, )
    counts = np.zeros(dims)
    for i in range(n_classes):
        counts[..., i] = (x==i).sum(axis=x.ndim-1)
    labels = counts.argmax(axis=x.ndim-1)
    return labels

def get_crosshairs(mask):
    """Determine center of mass of whole tumor for crosshairs plotting"""
    mask_bin = mask != 0
    xmax = mask_bin.sum((1,2)).argmax() + 1 # +1 for R indexing
    ymax = mask_bin.sum((0,2)).argmax() + 1
    zmax = mask_bin.sum((0,1)).argmax() + 1
    xyz = np.array([xmax, ymax, zmax])
    return xyz

def save_params(net, best_metric, current_metric, epoch, save_interval, prefix):
    """Logic for if/when to save/checkpoint model parameters"""
    if current_metric > best_metric:
        best_metric = current_metric
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_metric))
        with open(prefix+'_best.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_metric))
    if save_interval and (epoch + 1) % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_metric))
    return best_metric

def start_logger(args):
    """Start logging utilities for stdout, log files, and mxboard"""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))

    # Setup mxboard logging
    tb_dir = args.save_prefix + '_tb'
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    sw = SummaryWriter(logdir=tb_dir, flush_secs=60, verbose=False)

    return logger, sw

def log_epoch_hooks(epoch, train_loss, metrics, logger, sw):
    """Epoch logging"""
    DSCs = np.array([v['DSC'] for k,v in metrics.items()])
    DSC_avg = DSCs.mean()
    logger.info('E %d | loss %.4f | ET %.4f | TC %.4f | WT %.4f | Avg %.4f'%((epoch, train_loss) + tuple(DSCs) + (DSC_avg, )))
    sw.add_scalar(tag='Dice', value=('Val ET', DSCs[0]), global_step=epoch)
    sw.add_scalar(tag='Dice', value=('Val TC', DSCs[1]), global_step=epoch)
    sw.add_scalar(tag='Dice', value=('Val WT', DSCs[2]), global_step=epoch)
    sw.add_scalar(tag='Dice', value=('Val Avg', DSCs.mean()), global_step=epoch)
    return DSC_avg

class SoftDiceLoss(gluon.loss.Loss):
    """Soft Dice loss for segmentation"""
    def __init__(self, axis=-1, smooth=0, eps=1e-6, weight=None, batch_axis=0, **kwargs):
        super(SoftDiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._smooth = smooth
        self._eps = eps

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        # import pdb; pdb.set_trace()
        pred = F.softmax(pred, self._axis)

        label = F.one_hot(label, 4).transpose((0,4,1,2,3))

        tp = pred * label
        tp = F.sum(tp, axis=(self._axis, self._batch_axis), exclude=True, keepdims=True)

        tot = pred + label
        tot = F.sum(tot, axis=(self._axis, self._batch_axis), exclude=True, keepdims=True)

        dsc = (2 * tp + self._smooth) / (tot + self._smooth + self._eps)

        return - F.sum(dsc, axis=self._batch_axis, exclude=True)


import math 
import torch 
from torchvision import transforms
import numpy as np 


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts=None, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
#         if type(opts) == dict:
#             opts = munchify(opts)
        self.input_size = [3,224,224]#opts.input_size
        self.input_space = 'RGB'
        self.input_range = [0,1]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        tfs = []
        if preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale))))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width)))

        if random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
#         tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


def detach(input):
    return input.detach().cpu().numpy()

def original_create_batches(frames_to_do, load_img_fn, tf_img_fn, batch_size=32):
    n = len(frames_to_do)
    if n < batch_size:
        print("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    print("Generating {} batches...".format(n // batch_size))
    batches = []
    frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx + batch_size, n)))
        batch_frames = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((len(batch_frames),) + tuple(tf_img_fn.input_size))
        for i, frame_ in enumerate(batch_frames):
            input_img = load_img_fn(frame_)
            input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)
    return batches

def create_batches(frames_to_do, DIM, batch_size=1, skip_norm=False):
    n = frames_to_do.shape[0]
    h, w = frames_to_do.shape[1:3]
    scale = 0.875
    input_size = [3, DIM, DIM]
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    input_range = [0, 1.]
    input_space = 'RGB'
    expand_size = int(math.floor(max(input_size) / scale))
    if w < h:
        ow = expand_size
        oh = int(expand_size * h / w)
    else:
        oh = expand_size
        ow = int(expand_size * w / h)


    tfs = []
    tfs.append(ToSpaceBGR(input_space == 'BGR'))
    tfs.append(ToRange255(max(input_range) == 255))
#     tfs.append(transforms.Normalize(mean=mean, std=std))
    tf = transforms.Compose(tfs)

    a = int((0.5 * oh) - (0.5 * float(input_size[1])))
    b = a + input_size[1]
    c = int((0.5 * ow) - (0.5 * float(input_size[2])))
    d = c + input_size[2]

    if n < batch_size:
        # logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    # logger.info("Generating {} batches...".format(n // batch_size))

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx + batch_size, n)))

        # <batch, h, w, ch> <0,255>
        batch_tensor = frames_to_do[frames_idx]

        pass_in = batch_tensor.permute(0, 3, 1, 2) #/ 255.
        inp = torch.nn.functional.interpolate(pass_in,
                                              size=(oh, ow),
                                              mode='bilinear', align_corners=True)
        # Center cropping
        cropped_frames = inp[:, :, a:b, c:d]
        # cropped_image = cropped_image.contiguous()
        for i in range(len(cropped_frames)):
            cropped_frames[i] = tf(cropped_frames[i]) if not skip_norm else cropped_frames[i]
    return cropped_frames


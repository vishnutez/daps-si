from abc import ABC, abstractmethod
from piq import psnr, ssim, LPIPS
import prettytable
import torch
import torch.nn as nn
import wandb
import numpy as np
import warnings


class Evaluator:
    """
        Evaluation module for computing evaluation metrics.
    """

    def __init__(self, eval_fn_list):
        """
            Initializes the evaluator with the ground truth and measurement.

            Parameters:
                eval_fn_list (tuple): List of evaluation functions to use.
        """
        super().__init__()
        self.eval_fn = {}
        for eval_fn in eval_fn_list:
            self.eval_fn[eval_fn.name] = eval_fn
        self.main_eval_fn_name = eval_fn_list[0].name

    def get_main_eval_fn(self):
        """
            return the first eval_fn by default
        """
        return self.eval_fn[self.main_eval_fn_name]

    def __call__(self, gt, measurement, x, reduction='mean', **kwargs):
        """
            Computes evaluation metrics for the given input.

            Parameters:
                x (torch.Tensor): Input tensor.
                reduction (str): Reduction method ('mean' or 'none').

            Returns:
                dict: Dictionary of evaluation results.
        """
        results = {}
        for eval_fn_name, eval_fn in self.eval_fn.items():
            results[eval_fn_name] = eval_fn(gt, measurement, x, reduction, **kwargs)
        return results

    def to_list(self, x):
        return x.cpu().detach().tolist()

    def report(self, gt, measurement, x, **kwargs):
        '''x: [N, B, C, H, W] or [B, C, H, W]'''
        if len(x.shape) == 4:
            x = x[None]
        result_dicts = {}

        # eval function
        broadcasted_shape = torch.broadcast_shapes(x.shape, gt.shape)
        x0_flatten = gt.expand(broadcasted_shape).flatten(0, 1)
        x_flatten = x.expand(broadcasted_shape).flatten(0, 1)
        y_flatten = measurement.expand((broadcasted_shape[0], *measurement.shape)).flatten(0, 1)

        for key, fn in self.eval_fn.items():
            value = fn(x0_flatten, y_flatten, x_flatten, reduction='none', **kwargs).reshape(broadcasted_shape[0], -1)
            result_dicts[key] = {
                'sample': self.to_list(value.permute(1, 0)),
                'mean': self.to_list(value.mean(0)),
                'std': self.to_list(value.std(0) if value.shape[0] != 1 else torch.zeros_like(value.mean(0))),
                'max': self.to_list(value.max(0)[0]),
                'min': self.to_list(value.min(0)[0]),
            }
        return result_dicts

    def display(self, result_dicts):
        table = Table('results')
        summary = {}
        for key in result_dicts.keys():
            value = ['{:.3f}'.format(v) for v in result_dicts[key][get_eval_fn_cmp(key)]]
            table.add_column(key, value)
            summary[key] = '{:.3f}'.format(np.mean(result_dicts[key][get_eval_fn_cmp(key)]))
        # for average
        table.add_row(['' for _ in result_dicts.keys()])
        table.add_row(['avg' for _ in result_dicts.keys()])
        table.add_row(summary.values())

        return table.get_string()

    def log_wandb(self, result_dicts, batch_size):
        for s in range(batch_size):
            log_dict = {key: result_dicts[key][get_eval_fn_cmp(key)][s] for key in result_dicts.keys()}
            wandb.log(log_dict)
        log_dict = {key: np.mean(result_dicts[key][get_eval_fn_cmp(key)]) for key in result_dicts.keys()}
        new_log_dict = {key + '_all': value for key, value in log_dict.items()}
        wandb.log(new_log_dict)
        return


class Table(object):
    def __init__(self, title=None, field_names=None):
        """
            title:          str
            field_names:    list of field names
        """
        self.table = prettytable.PrettyTable(title=title, field_names=field_names)

    def add_rows(self, rows):
        """
            rows: list of tuples
        """
        self.table.add_rows(rows)

    def add_row(self, row):
        self.table.add_row(row)

    def add_column(self, fieldname, column):
        self.table.add_column(fieldname=fieldname, column=column)

    def get_string(self):
        """
            a markdown format table
        """
        _junc = self.table.junction_char
        if _junc != "|":
            self.table.junction_char = "|"
        markdown = [row for row in self.table.get_string().split("\n")[1:-1]]
        self.table.junction_char = _junc
        return "\n" + "\n".join(markdown)

    def get_latex_string(self):
        # TODO: to be done in future
        pass


__EVAL_FN__ = {}
__EVAL_FN_CMP__ = {}


def register_eval_fn(name: str):
    def wrapper(cls):
        if __EVAL_FN__.get(name, None):
            if __EVAL_FN__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVAL_FN__[name] = cls
        __EVAL_FN_CMP__[name] = cls.cmp
        cls.name = name
        return cls

    return wrapper


def get_eval_fn(name: str, **kwargs):
    if __EVAL_FN__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVAL_FN__[name](**kwargs)


def get_eval_fn_cmp(name: str):
    return __EVAL_FN_CMP__[name]


class EvalFn(ABC):
    def norm(self, x):
        return (x * 0.5 + 0.5).clip(0, 1)

    @abstractmethod
    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        pass


@register_eval_fn('psnr')
class PeakSignalNoiseRatio(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        return psnr(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('ssim')
class StructuralSimilarityIndexMeasure(EvalFn):
    cmp = 'max'  # the higher, the better

    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        return ssim(self.norm(gt), self.norm(sample), data_range=1.0, reduction=reduction)


@register_eval_fn('lpips')
class LearnedPerceptualImagePatchSimilarity(EvalFn):
    cmp = 'min'  # the lower, the better

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.lpips_fn = LPIPS(replace_pooling=True, reduction='none')

    def evaluate_in_batch(self, gt, pred):
        batch_size = self.batch_size
        results = []
        for start in range(0, gt.shape[0], batch_size):
            res = self.lpips_fn(self.norm(gt[start:start+batch_size]), self.norm(pred[start:start+batch_size]))
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res
    

# eval metrics for style transfer task
from clip.base_clip import CLIPEncoder
from torchvision import transforms

@register_eval_fn('style_loss')
class StyleLoss(EvalFn):
    cmp = 'min'  # the lower, the better

    def __init__(self, batch_size=128, res=224, device='cuda'):
        self.batch_size = batch_size
        self.clip_encoder = CLIPEncoder().cuda()  # trained-model for style transfer
        self.preprocess = transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        self.res = res
        self.device = device

    def compute_style_loss(self, image1, image2):

        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)

        image1 = torch.nn.functional.interpolate(image1, size=(self.res, self.res), mode='bicubic')
        image1 = self.preprocess(image1)

        image2 = torch.nn.functional.interpolate(image2, size=(self.res, self.res), mode='bicubic')
        image2 = self.preprocess(image2)
        
        f1, feats1 = self.clip_encoder.clip_model.encode_image_with_features(image1)
        f2, feats2 = self.clip_encoder.clip_model.encode_image_with_features(image2)
        
        feat1 = feats1[2][1:, :, :]
        feat2 = feats2[2][1:, :, :]

        feat1_mat = feat1.permute(1, 0, 2)
        feat2_mat = feat2.permute(1, 0, 2)

        feat1_mat_t = feat1_mat.permute(0, 2, 1)
        feat2_mat_t = feat2_mat.permute(0, 2, 1)

        # multiply the two matrices broadcasting the dimension
        gram1 = torch.bmm(feat1_mat_t, feat1_mat)
        gram2 = torch.bmm(feat2_mat_t, feat2_mat)

        return torch.linalg.norm(gram1 - gram2, axis=(-1, -2))  # dim batch size


    def evaluate_in_batch(self, gt, pred):
        batch_size = self.batch_size
        results = []
        for start in range(0, gt.shape[0], batch_size):
            res = self.compute_style_loss(gt[start:start+batch_size], pred[start:start+batch_size])
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        res = self.evaluate_in_batch(gt, sample)
        if reduction == 'mean':
            res = res.mean()
        return res      

from clip.clip import clip

@register_eval_fn('clip_score')
class ClipScore(EvalFn):
    cmp = 'max'  # the higher, the better

    def __init__(self, batch_size=128, res=224, device='cuda'):
        self.batch_size = batch_size
        self.clip_encoder = CLIPEncoder().cuda()  # trained-model for style transfer
        self.preprocess = transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        self.res = res
        self.device = device

    
    def compute_clip_score(self, image, text, normalize=False):
        text = clip.tokenize(text).cuda()
        image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        image = self.preprocess(image)
        image_feature, _ = self.clip_encoder.clip_model.encode_image_with_features(image)  # (b, d) with d=512
        text_feature = self.clip_encoder.clip_model.encode_text(text)  # (d,) with d=512
        text_feature = text_feature.repeat(image.shape[0], 1)  # (b, d) with d=512
        # compute inner product
        if normalize:
            normalized_image_feature = image_feature / torch.norm(image_feature, dim=1, keepdim=True)  # (b, d)
            normalized_text_feature = text_feature / torch.norm(text_feature, dim=1, keepdim=True)  # (b, d)
            normalized_clip_score = torch.bmm(normalized_image_feature.unsqueeze(1), normalized_text_feature.unsqueeze(2))  # (b, 1, d) * (b, d, 1) = (b, 1, 1)
            normalized_clip_score = normalized_clip_score.squeeze(1).squeeze(1)  # (b,)
            return normalized_clip_score
        else:
            clip_score = torch.bmm(image_feature.unsqueeze(1), text_feature.unsqueeze(2))  # (b, 1, d) * (b, d, 1) = (b, 1, 1)
            clip_score = clip_score.squeeze(1).squeeze(1)  # (b,)
            return clip_score


    def evaluate_in_batch(self, gt, pred, **kwargs):
        text = kwargs.get('text', 'a knight holding his knife')
        normalize = kwargs.get('normalize', False)
        print('text:', text)
        batch_size = self.batch_size
        results = []
        for start in range(0, pred.shape[0], batch_size):
            res = self.compute_clip_score(image=pred[start:start+batch_size], text=text, normalize=normalize)
            results.append(res)
        results = torch.cat(results, dim=0)
        return results

    def __call__(self, gt, measurement, sample, reduction='none', **kwargs):
        res = self.evaluate_in_batch(gt, sample, **kwargs)
        if reduction == 'mean':
            res = res.mean()
        return res      

    
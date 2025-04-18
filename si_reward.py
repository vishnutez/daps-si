from abc import ABC, abstractmethod
import torch
import numpy as np
from torchvision import transforms
from AdaFace import inference
from AdaFace.face_alignment import align, mtcnn
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from typing import Tuple, List
from clip import clip

__REWARD_METHOD__ = {}


def register_reward_method(name: str):
    def wrapper(cls):
        if __REWARD_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __REWARD_METHOD__[name] = cls
        return cls

    return wrapper


def get_reward_method(name: str, **kwargs):
    if __REWARD_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __REWARD_METHOD__[name](**kwargs)


class Reward(ABC):
    """
    Abstract base class for all reward functions used in guided diffusion or sampling.

    Subclasses should implement custom logic to compute a reward signal for a given input
    (e.g., image, text, or other modality) that can be used for steering diffusion sampling
    via gradient-based or search-based methods.

    Note:
        This base class is designed to be flexible for multiple types of guidance:
        - Face recognition similarity
        - Style transfer alignment
        - Text-to-image alignment
        - Any task-specific reward signal

    To implement a custom reward:
        1. Inherit from this class.
        2. Implement the `get_reward` method.
        3. Optionally implement any setup methods like `set_gt_embeddings`.

    Methods:
        get_reward(**kwargs): Abstract method to compute and return a reward score.
    """

    def __init__(self, **kwargs):
        """
        Optional constructor for reward classes. Accepts arbitrary keyword arguments
        for flexibility and downstream configuration.
        """
        pass

    @abstractmethod
    def get_reward(self, particles, **kwargs) -> torch.Tensor:
        """
        Compute and return the reward signal given inputs.

        Args:
            particles: The particles that you want to find the reward for
            **kwargs: Task-specific keyword arguments such as 'images', 'text', etc.

        Returns:
            A torch.Tensor representing the reward(s).
        """
        pass

    @abstractmethod
    def get_gradients(self, particles, **kwargs) -> torch.Tensor:
        """
        Compute and return the gradient of the difference of the embedding
        of particles with the embedding of given information.

        Args:
            particles: The particles that you want to find the gradient with respect to
            **kwargs: Task-specific keyword arguments

        Returns:
            A torch.Tensor representing the reward(s).
        """
        pass

    @abstractmethod
    def set_gt_embeddings(self, **kwargs):
        pass


@register_reward_method('adaface')
class AdaFaceReward(Reward):
    """
    Reward function based on AdaFace facial embeddings.

    This class computes the similarity between a generated face image and a reference
    face (additional image of the same person) using embeddings from a pretrained AdaFace model.

    The reward can be used for guiding diffusion models in tasks like face reconstruction,
    identity-preserving generation, and image alignment.

    Attributes:
        files (List[Path]): List of all image file paths in the dataset directory.
        model: Pretrained AdaFace embedding model.
        gt_embeddings (torch.Tensor): Ground-truth face embedding.
        device (str): Computation device, e.g., 'cuda' or 'cpu'.
        mtcnn_model: Face detector and aligner (MTCNN).
        res (int): Target image resolution for preprocessing.
    """

    def __init__(self, data_path: str, pretrained_model: str, resolution: int = 256, device: str = 'cuda:0', scale=1,
                 freq=1, **kwargs):
        """
        Initializes the AdaFaceReward class.

        Args:
            data_path (str): Path to the directory containing face images.
            pretrained_model (str): Name of the pretrained AdaFace model to load.
            resolution (int): Target resolution to resize and center crop images.
            device (str): Torch device for inference, default is 'cuda:0'.
            **kwargs: Additional unused keyword arguments.
        """
        super().__init__(**kwargs)
        file_types = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.device = device
        self.files = sorted([f for ft in file_types for f in Path(data_path).rglob(ft)])
        self.model = inference.load_pretrained_model(pretrained_model).to(self.device)
        self.gt_embeddings = None
        self.mtcnn_model = mtcnn.MTCNN(device=self.device, crop_size=(112, 112))
        self.res = resolution
        self.scale = scale
        self.freq = 1
        self.name = 'adaface'

    def get_reward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the negative L2 distance between the embeddings of given images
        and the stored ground-truth embedding.

        Args:
            images (torch.Tensor): Input batch of images (B, C, H, W) in [-1, 1].

        Returns:
            torch.Tensor: A tensor of shape B containing reward values.
        """
        embed = self._embeddings(images)
        return - torch.norm(self.gt_embeddings - embed, dim=1)

    def set_gt_embeddings(self, index: int) -> None:
        """
        Sets the ground-truth embedding by loading and embedding the additional image
        at the given index in the dataset.

        Args:
            index (int): Index of the reference image in the dataset list.
        """
        # Load and preprocess image
        img = Image.open(self.files[index])
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.res),
            transforms.CenterCrop(self.res)
        ])
        img_tensor = (trans(img) * 2 - 1).to(self.device)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.expand(3, -1, -1)

        # Set gt embedding
        self.gt_embeddings = self._embeddings(img_tensor.unsqueeze(0)).detach()

    def _embeddings(self, tensor_images: torch.Tensor) -> torch.Tensor:
        """
        Computes AdaFace embeddings for a batch of images.

        Each image is aligned using MTCNN and passed through the pretrained model.

        Args:
            tensor_images (torch.Tensor): Batch of images (B, C, H, W) in [-1, 1].

        Returns:
            torch.Tensor: A tensor of shape (B, D) with D-dimensional embeddings.
        """
        tensor_images = ((tensor_images + 1) / 2 * 255).clamp(0, 255).byte()
        to_pil = transforms.ToPILImage()

        aligned_images, failed_indices = [], []
        for i in range(tensor_images.size(0)):
            try:
                img = to_pil(tensor_images[i])
                aligned = align.get_aligned_face('', rgb_pil_image=img)
                aligned_images.append(inference.to_input(aligned).to(self.device))
            except Exception as e:
                print('Error in face alignment at index {0}, adding fallback embedding.'.format(i), flush=True)
                failed_indices.append(i)
                aligned_images.append(torch.randn((1, 3, 112, 112), device=self.device))

        batch_input = torch.cat(aligned_images, dim=0)  # Assuming dim=0 is batch
        embeddings, _ = self.model(batch_input)
        if failed_indices:
            fallback = torch.ones((len(failed_indices), self.gt_embeddings.shape[1]), device=embeddings.device) * 1e3
            embeddings[torch.tensor(failed_indices, device=embeddings.device)] = fallback

        return embeddings

    def get_gradients(self, images: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        images : (B, C, H, W) tensor in [-1, 1] **with** requires_grad=True.

        Returns
        -------
        distances : (B,)  L2 distances to `self.gt_embeddings`
        grads      : (B, C, H, W)  ∂distance/∂pixel  (same device as images)
        """

        B, C, H, W = images.shape
        images = images.clone().detach().requires_grad_(True)

        # ------------------------------------------------------------------
        # 1. Detect faces (no grad)
        # ------------------------------------------------------------------
        to_pil = transforms.ToPILImage()
        bboxes, failed = [], []

        for i in range(B):
            img_uint8 = ((images[i].detach() + 1) * 127.5).clamp(0, 255).byte().cpu()
            pil_img = to_pil(img_uint8)
            boxes, _ = self.mtcnn_model.align_multi(pil_img, limit=1)

            if len(boxes) == 0:  # fallback → use whole frame
                failed.append(i)
                print(30 * '*', flush=True)
                print(30 * '*', flush=True)
                print(30 * '*', flush=True)
                print('we are in get_gradients of AdaFace - len of boxes was 0 - does it mean face was not detected?',
                      flush=True)
                print(30 * '*', flush=True)
                print(30 * '*', flush=True)
                print(30 * '*', flush=True)
                bboxes.append(None)
            else:
                x1, y1, x2, y2 = boxes[0][:4].astype(int)
                # Clamp to valid range
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, W - 1), min(y2, H - 1)
                bboxes.append((x1, y1, x2, y2))

        # ------------------------------------------------------------------
        # 2. Differentiable crop → ( B , 3 , 112 , 112 )
        # ------------------------------------------------------------------
        face_tensors = []
        for i, bb in enumerate(bboxes):
            if bb is None:
                crop = torch.zeros((1, 3, 112, 112), device=images.device)
                print('returning zero gradient for no face', flush=True)
            else:
                x1, y1, x2, y2 = bb
                crop = images[i: i + 1, :, y1: y2 + 1, x1: x2 + 1]  # keeps grad
                crop = F.interpolate(crop, size=(112, 112),
                                     mode='bilinear', align_corners=False)
            face_tensors.append(crop)

        faces = torch.cat(face_tensors, dim=0)  # (B, 3, 112, 112)

        # ------------------------------------------------------------------
        # 3. Embeddings
        # ------------------------------------------------------------------
        embeds, _ = self.model(faces)  # (B, D)

        # ------------------------------------------------------------------
        # 4. L2 distance to reference embedding
        # ------------------------------------------------------------------
        if self.gt_embeddings is None:
            raise RuntimeError("Call set_gt_embeddings(...) first.")
        # distances = torch.norm(embeds - self.gt_embeddings, dim=1)  # (B,)
        distances = ((embeds - self.gt_embeddings) ** 2).sum(dim=1)

        # ------------------------------------------------------------------
        # 5. Back‑prop to get ∂distance/∂image
        # ------------------------------------------------------------------
        images.grad = None  # clear old grads
        distances.sum().backward()
        grads = images.grad.detach()  # (B, C, H, W)

        return grads


@register_reward_method('measurement')
class MeasurementReward(Reward):
    def __init__(self, scale=1, freq=1, **kwargs):
        super().__init__(**kwargs)
        self.operator = None
        self.scale = scale
        self.freq = 1
        self.name = 'measurement'

    def get_reward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.norm(kwargs.get('measurements') - self.operator.measure(images), p=2, dim=(1, 2, 3))

    def get_gradients(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.operator.gradient(images, kwargs.get('measurements'), return_loss=True)

    def set_operator(self, operator):
        self.operator = operator

    def set_gt_embeddings(self, index: int, **kwargs):
        pass


@register_reward_method('style')
class StyleReward:
    def __init__(self, model, data_path: str, device: str = 'cuda:0', scale=1,
                 freq=1, **kwargs):
        super().__init__(**kwargs)

        from clip.base_clip import CLIPEncoder

        self.model = model  # diffusion model for decoding the latents
        self.clip_encoder = CLIPEncoder().cuda()  # trained-model for style transfer
        file_types = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.device = device
        self.files = sorted([f for ft in file_types for f in Path(data_path).rglob(ft)])
        self.gt_embeddings = None
        self.scale = scale
        self.freq = 1
        self.name = 'clip'
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_reward(self, latents):  # images (B, C, H, W) in [-1, 1]
        images = self.model.decode(latents)
        embd_gram = self._embd_gram(images)
        return -torch.linalg.norm(self.gt_embd_gram - embd_gram, axis=(-1, -2))


    def set_gt_embd_gram(self, index: int):
        """
        Sets the ground-truth embedding by loading and embedding the additional image
        at the given index in the dataset.

        Args:
            index (int): Index of the reference image in the dataset list.
        """
        # Load and preprocess image

        img = Image.open(self.files[index]).convert('RGB')
        image = img.resize((224, 224), Image.Resampling.BILINEAR)
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()

        # Set gt embedding
        self.gt_embd_gram = self._embd_gram(img).detach()  # gram matrix of the embd

    def _embd_gram(self, tensor_images):

        emb, feats = self.clip_model.encode_image_with_features(tensor_images)
        feat = feats[2][1:, :, :]  # get the last layer features
        feat = feat.permute(1, 0, 2)
        feat_T = feat.permute(0, 2, 1)  # T -> transpose
        feat_gram_mat = torch.bmm(feat_T, feat)  # multiply the two matrices broadcasting the dimension

        return feat_gram_mat
    
    def get_gradients(self, latents: torch.Tensor, **kwargs):

        latents = latents.clone().detach().requires_grad_(True)
        rewards = self.get_reward(latents)
        rewards_grad = torch.autograd.grad(rewards.sum(), latents)[0]  # returns only the gradients

        return rewards_grad.detach()

        
        





"""Main script for MaskCLIP model."""

import hashlib
import os
import urllib
import warnings
from typing import Any, Callable, Optional, Union, List, Tuple
from pkg_resources import packaging

import torch
from torch import nn, Tensor
import PIL
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .model import build_model, MaskCLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .utils import available_datasets, get_classes

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = [
    "available_models",
    "available_datasets",
    "load",
    "tokenize",
    "generate_text_embeddings",
]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


_BASE_PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

_MASKCLIP_PROMPT_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
    "there is a {} in the scene.",
    "there is the {} in the scene.",
    "this is a {} in the scene.",
    "this is the {} in the scene.",
    "this is one {} in the scene.",
]


def _download(url: str, root: str) -> str:
    """Download model.

    Parameters
    ----------
    url : str
        Link to model file.
    root : str
        Path where to save model.

    Returns
    -------
    str
        Path to downloaded file.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _convert_image_to_rgb(image: PIL.Image) -> PIL.Image:
    """Convert image file.

    Parameters
    ----------
    image : PIL.Image
        Image file.

    Returns
    -------
    PIL.Image
        Converted to RGB image.
    """
    return image.convert("RGB")


def _transform(n_px: int) -> Callable:
    """Return transforms.

    Parameters
    ----------
    n_px : int
        Number of pixels for resizing image.

    Returns
    -------
    Callable
        Transforms.
    """
    return T.Compose(
        [
            T.Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda:0" if torch.cuda.is_available() else "cpu",
    input_resolution: int = None,
    dataset: str = None,
    text_feature_layer: int = -1,
    templates: str = "maskclip",
    **kwargs: Any,
) -> Tuple[nn.Module, Callable]:
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `maskclip.available_models()`, or the path to a model checkpoint containing the state_dict.
    device : Union[str, torch.device]
        The device to put the loaded model.
    input_resolution : int
        The `input_resolution` of model.
    dataset : str
        A dataset name if needed initialization of `text_embeddings` for MaskCLIP.
    text_feature_layer : int
        Text feature layer/transformer layer for extraction of text embeddings.
    templates : str
        Prompt templates for generation of text embeddings.

    Returns
    -------
    model : torch.nn.Module
        The MaskCLIP model.
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/maskclip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), **kwargs).to(device)
    if str(device) == "cpu":
        model.float()

    if input_resolution and input_resolution != model.visual.input_resolution:
        model = model.eval()
        model.visual.resize_positional_embedding(input_resolution)
    if dataset is not None:
        assert dataset in available_datasets(), ValueError(
            f"unknown dataset: {dataset}"
        )
        classnames = get_classes(dataset)
        text_embeddings = generate_text_embeddings(
            model,
            classnames,
            device=device,
            templates=templates,
            feature_layer=text_feature_layer,
        )
        if str(device) != "cpu":
            text_embeddings = text_embeddings.half()
        model.initialize_text_embeddings(text_embeddings)
    return model, _transform(input_resolution or model.visual.input_resolution)


def tokenize(
    texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False
) -> Union[torch.IntTensor, torch.LongTensor]:
    """Return the tokenized representation of given input string(s).

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize.
    context_length : int
        The context length to use; all CLIP models use 77 as the context length.
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length.

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, : len(tokens)] = torch.tensor(tokens)

    return result


@torch.no_grad()
def generate_text_embeddings(
    model: torch.nn.Module,
    classnames: List[str],
    templates: Union[List[str], str] = "maskclip",
    device: Union[str, torch.device] = "cuda:0" if torch.cuda.is_available() else "cpu",
    feature_layer: int = None,
    context_length: int = 77,
) -> torch.Tensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    model : torch.nn.Module
        A MaskCLIP model from `maskclip.available_models()`
    classnames : List[str]
        The list of dataset class names to generate text embeddings
    templates : Optional[List[str]]
        The list of prompt templates to generate text embeddings. If `None` default prompts are used
    device : Union[str, torch.device]
        The device to put the loaded model
    input_resolution: int
        Input resolution for transformer if different from model`s size is required

    Returns
    -------
    A two-dimensional tensor containing the text embeddings for certain dataset
    """
    model = model.to(device)
    model.eval()
    if isinstance(templates, str):
        prompt_mapping = {
            "base": _BASE_PROMPT_TEMPLATES,
            "maskclip": _MASKCLIP_PROMPT_TEMPLATES,
        }
        assert templates in prompt_mapping.keys()
        templates = prompt_mapping[templates]

    if "background" in classnames:
        idx = classnames.index("background")
        classnames.pop(idx)

    text_embeddings = []
    for classname in classnames:
        if templates is None:
            texts = [classname]
        else:
            texts = [
                template.format(classname) for template in templates
            ]  # format with class
        texts = tokenize(texts, context_length=context_length).to(device)  # tokenize
        class_embeddings = model.encode_text(
            texts, feature_layer=feature_layer
        )  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_embeddings.append(class_embedding)
    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)
    return text_embeddings.permute(1, 0).float()

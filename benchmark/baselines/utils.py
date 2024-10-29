import re
import torch


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def torch_default_device():
    """
    Determine which device to use for calculations automatically.

    Notes: MPS is prioritized if available for the case where code is running on a M* MacOs device.

    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

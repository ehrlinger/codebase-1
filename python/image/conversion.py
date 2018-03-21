import numpy as np
from PIL import Image



def image_pil2NumPy_array(img, scale = True, channel_first=True):
    """Convert a PIL image to a numpy image.
    Args:
        img (PIL image): A PIL image of uint8 between 0 and 255 using RGB channels.
    Returns:
        img_new (numpy array): A numpy image of uint8 between 0 and 255.
    Examples:
        >>> img = Image.open('../../share/automobile10.png')
        >>> img_conv = np.arry(img)
        >>> img_conv.shape
        (32, 32, 3)
        >>> img_conv = img_conv.reshape(-1, 3, 32, 32)
        >>> img_conv.shape
        (1, 3, 32, 32)
        
    """

    img_conv = np.array(img)
    # Scale pixel intensity
    if scale:
        img_conv = img_conv / 255.0

    # Reshape
    img_conv = img_conv.reshape(-1, 3, 32, 32)
    
    # Channel last
    if not channel_first:
        img_conv = np.swapaxes(img_conv, 1, 3)
       
    return img_conv

     
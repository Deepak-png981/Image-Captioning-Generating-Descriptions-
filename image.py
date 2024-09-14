import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the image from a URL
# img_url = 'https://images.pexels.com/photos/3628912/pexels-photo-3628912.jpeg?cs=srgb&dl=pexels-case-originals-3628912.jpg&fm=jpg'
# img = Image.open(requests.get(img_url, stream=True).raw)
img_url = 'image.jpg'
img = Image.open(img_url)
# Process the image and generate a caption
inputs = processor(img, return_tensors="pt")
caption = model.generate(**inputs)

# Decode and print the generated caption (without extension warnings)
print(processor.decode(caption[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))




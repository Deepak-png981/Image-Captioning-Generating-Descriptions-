import warnings
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the FastAPI app
app = FastAPI()

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Process the image and generate a caption
    inputs = processor(img, return_tensors="pt")
    caption = model.generate(**inputs)
    
    # Decode the generated caption
    caption_text = processor.decode(caption[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Return the caption as a response
    return {"caption": caption_text}


from PIL import Image, ImageDraw, ImageFont
import os

def create_signature(image_name, text, is_forged=False):
    # Create a blank image with white background
    img = Image.new('RGB', (200, 100), color='white')
    d = ImageDraw.Draw(img)

    # Use a basic font
    font = ImageFont.load_default()

    # Draw text on the image
    d.text((10, 40), text, fill='black', font=font)

    # Save the image
    img.save(os.path.join('static', 'reference_signatures', image_name))

# Create 10 genuine and forged signatures
for i in range(1, 11):
    create_signature(f"{i:03d}_real.png", f"Signature {i}")
    create_signature(f"{i:03d}_forged.png", f"Forged Signature {i}", is_forged=True)

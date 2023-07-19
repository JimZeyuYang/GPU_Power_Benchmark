from PIL import Image

def upscale_image(filename):
    # Load the image
    img = Image.open(filename)

    # Get the size of the image
    width, height = img.size

    # Calculate the new size
    ratio = 8
    new_size = (width*ratio, height*ratio)

    # Resize the image
    img_upscaled = img.resize(new_size, Image.BICUBIC)

    # Save the new image
    img_upscaled.save("upscaled_" + filename)

upscale_image('stereo.im0.640x533.ppm')
upscale_image('stereo.im1.640x533.ppm')

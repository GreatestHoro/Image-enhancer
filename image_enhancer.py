import cv2
from cv2 import dnn_superres
from tqdm import tqdm
import os 
import numpy as np

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()


# get image resolution

def get_image_resolution(input_path):
    print("image path", input_path)
    # Read image
    image = cv2.imread(input_path)
    height, width, channels = image.shape
    print("image read wiht size", image.shape)
    return image
    # Read the desired model



def read_model(path):
    sr.readModel(path)
    print("{} model read successfully".format(path))

def set_scale(name, scale):
    sr.setScale(scale)
    print("scale set to {}".format(scale))
# sr.setModel("edsr", 2)

def upscale(image, output_path):
    # Upscale the image
    result = sr.upsample(image)
    cv2.imwrite(output_path + "upscaled.jpg", result)



def remove_background(image, output_path):
    img = get_image_resolution(image)

    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

    # negate mask
    mask = 255 - mask

    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save resulting masked image
    cv2.imwrite(output_path, result)


def main():     
    root = os.getcwd()
    image_name = "giacometti-city-square.jpg"
    input_path = root + "/image/"
    output_path = root + "/output/"
    model = "x4.pb"
    model_path = root + "/model/" + model
    model_name = "edsr"
    scale = 4

    # image = get_image_resolution(input_path + image_name)
    # image = get_image_resolution(input_path)
    remove_background(input_path + image_name, output_path + image_name)
    # read_model(model_path)
    # set_scale(model_name, scale)
    # upscale(image, output_path)


main()
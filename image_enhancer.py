import cv2
from cv2 import dnn_superres
from tqdm import tqdm

input_path = 'C:/Users/ronih/Documents/fun/image/giacometti-city-square.jpg'
output_path = 'C:/Users/ronih/Documents/fun/output'
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# get image resolution



# Read image
image = cv2.imread(input_path)
height, width, channels = image.shape
print("image read wiht size", image.shape)
# Read the desired model

path = "C:/Users/ronih/Documents/fun/model/model_two.pb"
sr.readModel(path)
print("{} model read successfully".format(path))


# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 2)
# print ("model set")
# create a progressbar with tqdm to show progress
# for i in tqdm(range(1, 100)):
#     # Resize the image to the desired scale
#     image_resized = cv2.resize(image, (width // 2, height // 2))
#     # Perform super-resolution
#     image_superres = sr.upsample(image_resized)
#     # Save the image
#     cv2.imwrite(output_path + "/" + str(i) + ".jpg", image_superres)

# Upscale the image
result = sr.upsample(image)
print( "image upscaled with new resolution {}").format(result.shape)
# Save the image
cv2.imwrite(output_path + "upscaled.jpg", result)
import glob
import numpy as np
from config import CONFIG
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def parse_data():
    "Function that parses the train and test images and masks"
    
    train_images = glob.glob(CONFIG.IMG_PATH + "/*.jpg")
    train_mask=[]
    
    #Mask path is obtained from image path
    for i in range(len(train_images)):
        path_split = train_images[i].split(sep="/")
        path_split[3] = "masks"
        new_path = "/".join(path_split)
        path_split_2 = new_path.split(sep=".")
        path_split_2[3] = "png"
        final_path = ".".join(path_split_2)
        train_mask.append(final_path)

    #train and test split
    test_mask = train_mask[-CONFIG.NO_OF_TEST_SAMPLES:]
    train_mask = train_mask[:-CONFIG.NO_OF_TEST_SAMPLES]
    test_images = train_images[-CONFIG.NO_OF_TEST_SAMPLES:]
    train_images = train_images[:-CONFIG.NO_OF_TEST_SAMPLES]
    
    return train_images,train_mask ,test_images,test_mask 

def get_data(mode):
    "Function create image and mask arrays"
    
    
    print("Getting and Resizing Train Images, Train Mask, and Adding Label ...\n\n")
    
    #Get the train and test images and mask
    train_images,train_mask ,test_images,test_mask  = parse_data()
    
    if mode == "train":
        images = train_images
        masks = train_mask
    elif mode == "test":
        images = test_images
        masks = test_mask
        
    # Create image arrays
    X_seg = np.zeros((len(images), CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, CONFIG.IMG_CHANNEL_IMAGE), dtype=np.uint8)

    for i, data_row in tqdm(enumerate(images), total=len(images)):

        FaceImage = cv2.imread(data_row)
        FaceImage = cv2.resize(FaceImage, (CONFIG.IMG_WIDTH,CONFIG.IMG_HEIGHT))

        X_seg[i] = FaceImage

    print('\n\nProcess ... C O M P L E T E')

    #Create mask arrays
    Y_seg = np.zeros((len(masks), CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, CONFIG.IMG_CHANNEL_MASK), dtype=np.bool)
    for i, data_row in tqdm(enumerate(masks), total=len(masks)):

        FaceImage = cv2.imread(data_row,0)
        FaceImage = cv2.resize(FaceImage, (CONFIG.IMG_WIDTH,CONFIG.IMG_HEIGHT))
        FaceImage = np.expand_dims( FaceImage,axis=-1)

        Y_seg[i] = FaceImage

    plt.imshow(X__seg[0])
    plt.imshow(Y_seg[0])


    return X_seg,Y_seg

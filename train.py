import numpy as np
import pandas as pd
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from config import CONFIG
from metrics import bce_dice_loss,dice_coef,iou_coef
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from data_prepare import get_data


def create_train_model():
    "Function create Linknet model with efficientnetb7 backbone and does training"
    
    #create model
    model = Linknet('efficientnetb7',input_shape=(CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH, CONFIG.IMG_CHANNEL_MASK),
                                                  activation='sigmoid', encoder_weights='imagenet')
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef,iou_coef]) 

    checkpoint = ModelCheckpoint(
        'UNET_model',
        monitor='val_loss',
        verbose=1, 
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
    )

    early_stopping = EarlyStopping(
        patience=5,
        min_delta=0.0001,
        restore_best_weights=True,
    )
    
    #Get train data
    X_train_seg, Y_train_seg = get_data(mode = "train")

    history = model.fit(
        X_train_seg, Y_train_seg,
        validation_split=CONFIG.train_test_split,
        callbacks=[checkpoint, early_stopping],
        epochs=CONFIG.EPOCHS,
        batch_size=CONFIG.BATCH_SIZE
    )
    
    return model

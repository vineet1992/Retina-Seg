""" Train unet for CXR segmentation

Usage:
  train_unet.py <data_frame> <image_dir> <mask_dir> <model_dir> <model_name> [--epochs=EPOCHS] [--startmdl=STARTMDL]
  train_unet.py (-h | --help)
Examples:
  train_unet.py /path/to/data/frame.csv /path/to/images /path/to/write/output/model.pth
Options:
  -h --help                    Show this screen.
  --epochs==EPOCHS             Number of epochs to train [Default: 100]
  --startmdl==STARTMDL         Path of an existing model to load before running training [Default: None]

"""

###Assign training and validation indices -> Make forum post about transformations?
###Run this for a single epoch and validate that image output is working, output to directory instead

##TODO make age,sex more general instead of hardcoding it




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from docopt import docopt
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import *
from fastai.callbacks import *
import math
import time

from SegmentationUtils import SegmentationDataset
from SegmentationUtils import SegmentationModel
from SegmentationUtils import SoftDiceLoss
from SegmentationUtils import CrossEntropy

from AutoencoderUtils import BasicBlockBack
from AutoencoderUtils import UpSample

from Unet import *

from unet_model import UNet
from unet_parts import *


tfms_test = get_transforms(do_flip = False,max_warp = None)
tfms = get_transforms(do_flip = True, 
max_rotate = 5.0, max_zoom = 1.5, max_lighting=0.5,max_warp = None)
num_workers = 8
bs = 2
size = 320

if __name__ == '__main__':

    arguments = docopt(__doc__)

    ###Experiment Dataframe (expecting a column called Model_Location
    df = pd.read_csv(arguments['<data_frame>'])
  
    ###Grab image directory
    image_dir = arguments['<image_dir>']
    
    ###Path to output model
    model_dir = arguments['<model_dir>'] 
    model_name = arguments['<model_name>']

    model_path = os.path.join(model_dir,model_name)

    ### Number of epochs
    num_epochs = int(arguments['--epochs'])
    
    valid_idx = df.index[df.Dataset=="Tu"]
   
    train_masks = df.Filename[df.Dataset=="Tr"]
    valid_masks = df.Filename[df.Dataset=="Tu"]
    

    #imgs = (ImageList.from_df(df=df,path=image_dir)
    #                        #.split_none()
    #                        .split_by_idx(valid_idx)
    #                        .label_from_df(cols="Dummy")
    #                        .databunch(num_workers = num_workers,bs=bs))
                      
  
    train_ds = SegmentationDataset(train_masks,image_dir,arguments['<mask_dir>'],transforms=tfms,size=size)
    valid_ds = SegmentationDataset(valid_masks,image_dir,arguments['<mask_dir>'],transforms=tfms,size=size)
    
    
    #train_dl = DataLoader(train_ds, bs)
    #valid_dl = DataLoader(valid_ds, 2 * bs)
    data = DataBunch.create(train_ds, valid_ds, bs=bs, num_workers=num_workers,path=image_dir)
    #data = DataBunch(train_dl, valid_dl, path=image_dir)
    #import pdb; pdb.set_trace()
    model = UNet(3,1,bilinear=False)
    if(arguments['--startmdl']!="None"):
        model = UNet(3,3)
    
    tmp = data.train_ds[0][0]
    model(tmp.view(1,3,size,size))
    
    
    ###Create encoder based on simple convolutional architecture
    # enc = nn.Sequential()

    # enc.add_module("Conv1",nn.Conv2d(3,64,kernel_size=(7,7)))
    # enc.add_module("BN1",nn.BatchNorm2d(64))
    # enc.add_module("ReLU1",nn.ReLU())
    # enc.add_module("Max1",nn.MaxPool2d(3,stride=2))

    # enc.add_module("Conv2",nn.Conv2d(64,128,kernel_size=(7,7)))
    # enc.add_module("BN2",nn.BatchNorm2d(128))
    # enc.add_module("ReLU2",nn.ReLU())
    # enc.add_module("Max2",nn.MaxPool2d(3,stride=2))

    # enc.add_module("Conv3",nn.Conv2d(128,256,kernel_size=(7,7)))
    # enc.add_module("BN3",nn.BatchNorm2d(256))
    # enc.add_module("ReLU3",nn.ReLU())
    # enc.add_module("Max3",nn.MaxPool2d(3,stride=2))

    # enc.add_module("Conv4",nn.Conv2d(256,512,kernel_size=(7,7)))
    # enc.add_module("BN4",nn.BatchNorm2d(512))
    # enc.add_module("ReLU4",nn.ReLU())
    # enc.add_module("Max4",nn.MaxPool2d(3,stride=2))

    # enc.add_module("AvgPool",nn.AvgPool2d(7))

    # class Flatten(nn.Module):
        # def forward(self, x):
            # x = x.view(x.size()[0], -1)
            # return x
            
    # class Unflatten(nn.Module):
        # def forward(self, x):
            # x = x.view(x.size()[0], x.size()[1],1,1)
            # return x

    # enc.add_module("Flatten",Flatten())
   
    
    # ###Create simple convolutional decoder to go back to the image from the encoding
    # ###Create decoder
    
    # m = nn.Sequential()


    # ###Play with this to get an appropriate receptive field and to reconstruct the shape of the image properly TODO
    # m.add_module("Unflatten",Unflatten())
    # m.add_module("Back1",BasicBlockBack(512,256,(7,7),upsample=True))
    # m.add_module("Back2",BasicBlockBack(256,128,(14,14),upsample=True))
    # m.add_module("Back3",BasicBlockBack(128,64,(28,28),upsample=True))
    # m.add_module("Back4",BasicBlockBack(64,64,(56,56),upsample=True))
    # upSamp = UpSample(64,3,out_shape=(224,224),scale=None)
    # m.add_module("Final_Up",upSamp)
    # act_f = nn.Sigmoid()
    # m.add_module('Sigmoid',act_f)

  
    # ###Conv + sig to get to 3 x 224 x 224
    # #nba.add_layer(m,64,3,"FinalConv",(224,224),scale=None,act='sig')

    
    # model = SegmentationModel(enc,m)
    
    learn = Learner(data, model)
    learn.model_dir = model_dir
    
    if(arguments['--startmdl']!="None"):
        #Cut # of classes to 1
        learn.load(arguments['--startmdl'])
        learn.model.outc = OutConv(64,1)
        learn.model.cuda()
    

    #learn.loss_func = SoftDiceLoss()
    learn.loss_func = CrossEntropy()
    callbacks = [fastai.callbacks.tracker.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=model_path)]
    max_lr = 1e-3
    learn.fit_one_cycle(num_epochs,max_lr=max_lr,callbacks=callbacks)


    
    

""" Run Ensemble Retinal vessel segmentation

Usage:
  run_segment.py <image_dir> <mask_dir> <model_path> <num_models> [--cleanIso] [--color]
  run_segment.py (-h | --help)
Examples:
  train_unet.py /path/to/data/frame.csv /path/to/images /path/to/write/output/model.pth
Options:
  -h --help                    Show this screen.
  --cleanIso                   Should we clean isolated samples?
  --color                      Are the masks in color?
"""




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from docopt import docopt
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from sklearn.metrics import *
from fastai.callbacks import *
import math
import time

from SegmentationUtils import SegmentationDataset
from SegmentationUtils import SegmentationModel
from SegmentationUtils import SoftDiceLoss

from torchvision.utils import save_image

from unet_model import UNet
import cv2
from scipy.ndimage import label, generate_binary_structure


tfms_test = get_transforms(do_flip = False,max_warp = None,max_rotate=0.0,max_zoom=1.0,max_lighting=0.0)
tfms = get_transforms(do_flip = False, 
max_rotate = 5.0, max_zoom = 1.2, max_lighting=0.5,max_warp = None)
num_workers = 8
bs = 64
size = 320
cont_thresh = 50
thresh = 1



if __name__ == '__main__':

    arguments = docopt(__doc__)
    
    ##Grab image directory
    image_dir = arguments['<image_dir>']
    mask_dir = arguments['<mask_dir>']
    if(not os.path.exists(mask_dir)):
        os.mkdir(mask_dir)
    output_df = pd.DataFrame()
    output_df['Files'] = os.listdir(image_dir)
    output_df['Dummy'] = np.random.rand(len(output_df['Files']))
    col = "Dummy"
###Path to output model
    model_path = arguments['<model_path>'] 

    

    imgs = (ImageList.from_df(df=output_df,path=image_dir)
                            .split_none()
                            #.split_by_idx(valid_idx)
                            .label_from_df(cols=col)
                            .databunch(num_workers = num_workers,bs=bs))
                          
    learners = []
    num_models = int(arguments['<num_models>'])

    for i in range(num_models):
        model = UNet(3,1,bilinear=False)
        learn = Learner(imgs, model)
        learn.model_dir = ""
        learn.load(arguments['<model_path>'] + "_" + str(i))
        
        learn.model.eval()
        learn.model.cuda()
        learners.append(learn.model)

    dice = []
    area = np.zeros((output_df.shape[0],2)) #Est, Actual, Arteries, Veins
    fd = np.zeros((output_df.shape[0],2))
    
    
    pbar = progress_bar(range(output_df.shape[0]))

    for i in pbar:
        all_imgs = torch.zeros(num_models,3,size,size)
        #Load images output by each model
        for j in range(num_models):
            #import pdb; pdb.set_trace()

            #Read original input image
            curr = open_image(os.path.join(image_dir,output_df.iloc[i,0]))
            
            #Transform to correct size          
            curr = curr.resize(size).px
            
            #Apply segmentation model j
            mask = learners[j](curr.view(1,3,size,size).cuda())
            
            mask = 1-torch.round(mask*thresh)

            mask = torch.mode(mask,dim=1)
            tmp = mask.values.cpu().detach().numpy()
            mask_for_dice = 1-tmp
            if(arguments['--cleanIso']):
                arr,nf = label(tmp)
                unique,counts = np.unique(arr.flatten(),return_counts=True)
                lbls_keep = unique[counts < cont_thresh]
                tmp[np.isin(arr,lbls_keep)] = 0
                mask = torch.Tensor(tmp)
            else:
                mask = mask.values.cpu().detach()

            mask = mask.repeat(1,3,1,1)
            
            
            mask = 1 - mask
            #Store each image
            all_imgs[j,:,:,:] = mask.clone().detach().cpu()
            mask = mask*curr.view(1,3,size,size)
                        

        
        #Majority vote for pixel output
        final_mask = torch.mean(all_imgs,0)
        final_mask = torch.round(final_mask*thresh)
               
        

        save_image(final_mask.view(3,size,size).clone().detach().cpu(),os.path.join(mask_dir,output_df.iloc[i,0]))

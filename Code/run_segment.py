""" Run CXR segmentation - heart

Usage:
  run_segment.py <image_dir> <mask_dir> <model_path> [--dataframe=DF] [--target=TARGET] [--split=SPLIT] [--checkFiles] [--trueDir=TD] [--cleanIso] [--output_df=DF] [--color] [--invert]
  run_segment.py (-h | --help)
Examples:
  train_unet.py /path/to/data/frame.csv /path/to/images /path/to/write/output/model.pth
Options:
  -h --help                    Show this screen.
  --dataframe=DF               Optional data frame to select which images are of interest [default: None]
  --target=TARGET              If optional df is specified, then need to include the target variable [default: None]
  --split=SPLIT                If split, then split on the Dataset column keeping only the Te values [default: False]
  --checkFiles                 Should we check whether df files actually exist?
  --trueDir=TD                 Directory of ground-truth masks [Default:None]
  --cleanIso                   Should we clean isolated samples?
  --output_df=DF               Output dataframe to file? [Default:None]
  --color                      Are the masks in color?
  --invert                     Does the mask need to be inverted?
"""




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


def DICE(im1,im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

if __name__ == '__main__':

    arguments = docopt(__doc__)
    
    ##Grab image directory
    image_dir = arguments['<image_dir>']
    mask_dir = arguments['<mask_dir>']
    if(not os.path.exists(mask_dir)):
        os.mkdir(mask_dir)
    
    if(arguments['--dataframe']=="None"):
        files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
        ###Results
        output_df = pd.DataFrame(columns = ['File','Dummy','Prediction'])
        
        output_df['File'] = files
        output_df['Dummy'] = np.random.random_sample(len(files))
        col = 'Dummy'
    else:
        output_df = pd.read_csv(arguments['--dataframe'])
        locs = []
        if(arguments['--checkFiles']):
            for i in range(0,output_df.shape[0]):
                if(os.path.exists(os.path.join(image_dir,output_df.iloc[i,0]))):
                    locs.append(i)
                else:
                    print(output_df.iloc[i,0])
            output_df = output_df.iloc[locs,:]
        
            output_df = output_df.reset_index(drop=True)  
        col = arguments['--target']
        
        if(arguments["--split"]!="False"):
            output_df = output_df[output_df.Dataset=="Te",]
  

    
    ###Path to output model
    model_path = arguments['<model_path>'] 

    

    imgs = (ImageList.from_df(df=output_df,path=image_dir)
                            .split_none()
                            #.split_by_idx(valid_idx)
                            .label_from_df(cols=col)
                            .databunch(num_workers = num_workers,bs=bs))
                          

    model = UNet(3,1,bilinear=False)
    if(arguments['--color']):
        model = UNet(3,3)

    
    learn = Learner(imgs, model)
    
    learn.load(arguments['<model_path>'])
    learn.model.eval()
    learn.model.cuda()

    all_dice = []
    all_roc = []
    
    #import pdb; pdb.set_trace()
    pbar = progress_bar(range(output_df.shape[0]))
    for i in pbar:
        #import pdb; pdb.set_trace()
        curr = open_image(os.path.join(image_dir,output_df.iloc[i,0]))
        curr = curr.apply_tfms(tfms_test[0],size=size,do_resolve=False).px        
        mask = learn.model(curr.view(1,3,size,size).cuda())
        

        if(arguments['--color']):
            mask = torch.round(mask*thresh)
            save_image(mask.view(3,size,size).detach().cpu(),os.path.join(mask_dir,output_df.iloc[i,0]))
        else:


            if(arguments['--invert']):
                mask = torch.round(mask*thresh)
            else:
                mask = 1-torch.round(mask*thresh)
            #mask = 1-torch.round(mask)
            #kernel = np.ones((5,5),np.uint8)
            #erosion = cv2.erode(img,kernel,iterations = 1)
            #remove random dots - MORPH_CLOSE to fill in blanks
            #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            mask = torch.mode(mask,dim=1)
            #import pdb; pdb.set_trace()
            #img = mask.values.data.cpu().numpy()
            #kernel = np.ones((3,3),np.uint8)
            #opening = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
            #mask = torch.Tensor(opening)
            tmp = mask.values.cpu().detach().numpy()
            mask_for_dice = 1-tmp
            #import pdb; pdb.set_trace()
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
            mask = mask*curr.view(1,3,size,size)
            
            #Need to apply mask
            
            save_image(mask.view(3,size,size).detach().cpu(),os.path.join(mask_dir,output_df.iloc[i,0]))
            #import pdb; pdb.set_trace()
            #Add DICE score for to output df 
            if(not arguments['--trueDir'] is None and arguments['--trueDir']!="None" ):

                #Load ground truth mask
                curr2 = open_image(os.path.join(arguments['--trueDir'],output_df.iloc[i,0]))
                curr2 = curr2.apply_tfms(tfms_test[0],size=size,do_resolve=False).px
                curr2 = torch.round(curr2)
                curr2 = torch.mode(curr2,dim=0)
                tmp = curr2.values.cpu().detach().numpy()
                
                #Calculate intersection/union against predicted mask
                curr_dice =  DICE(tmp,mask_for_dice[0,:,:])
                
                #Calculate roc AUC
                curr_roc = roc_auc_score(tmp.flatten(),mask_for_dice[0,:,:].flatten())
                #import pdb; pdb.set_trace()

                #Add to vector
                all_dice.append(curr_dice)
                all_roc.append(curr_roc)

    if(arguments['--output_df']!="None"):
        output_df['DICE'] = pd.Series(all_dice)
        output_df['ROC'] = pd.Series(all_roc)
        output_df.to_csv(arguments['--output_df'])
        

    
    

""" Run CXR segmentation - heart

Usage:
  run_segment.py <image_dir> <mask_dir> <model_path> <num_models> <output_dir> [--dataframe=DF] [--target=TARGET] [--split=SPLIT] [--checkFiles] [--trueDir=TD] [--cleanIso] [--output_df=DF] [--color] [--invert]
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


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])

#Calculate fractal dimension
def calcFD(img,threshold=0.9):
    assert(len(img.shape)==2)
    # Minimal dimension of image
    p = min(img.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(img, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

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
    output_dir = arguments['<output_dir>']
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
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
            
            if(arguments['--invert']):
                mask = torch.round(mask*thresh)
            else:
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
            
            #Save image to file
            out_dir = mask_dir[:-1] + "_" + str(j)
            if(not os.path.exists(out_dir)):
                os.mkdir(out_dir)
            save_image(mask.view(3,size,size).clone().detach().cpu(),os.path.join(out_dir,output_df.iloc[i,0]))
            

        
        #Majority vote for pixel output
        final_mask = torch.mean(all_imgs,0)
        final_mask = torch.round(final_mask*thresh)
               
        

        save_image(final_mask.view(3,size,size).clone().detach().cpu(),os.path.join(output_dir,output_df.iloc[i,0]))
        if(not arguments['--trueDir'] is None and arguments['--trueDir']!="None" ):
            #Load ground truth mask
            curr2 = open_image(os.path.join(arguments['--trueDir'],output_df.iloc[i,0]))
            curr2 = curr2.resize(size).px
            curr2 = torch.round(curr2)
            curr2 = torch.mode(curr2,dim=0)
            tmp = curr2.values.cpu().detach().numpy()
            
            #Calculate intersection/union against predicted mask
            curr_dice =  DICE(tmp,final_mask.detach().numpy()[0,:,:])

            
            #Add to vector
            dice.append(curr_dice)            
            #Estimated arteries and veins
            est = np.round(final_mask.detach().numpy()[0,:,:])
            
            #Compute total area
            area[i,0] = np.sum(est)
            area[i,1] = np.sum(np.round(tmp))
            
            #Compute fractal dimension
            fd[i,0] = calcFD(est)
            fd[i,1] = calcFD(np.round(tmp))
            
            
            # for j in range(num_models):
                # guess = rounded_masks.clone().detach().cpu()
                # guess = torch.round(guess*thresh).numpy()
                
                # seg_guess = np.amax(guess,0)
                # seg_actual = np.amax(tmp,0)
                # tp = np.sum(np.logical_and(seg_guess,seg_actual))
                # fp = np.sum(np.logical_and(seg_guess,1-seg_actual))
                # fn = np.sum(np.logical_and(1-seg_guess,seg_actual))
                # precision[i,j] = tp / (tp + fp)
                # recall[i,j] = tp / (tp + fn)
            

    if(arguments['--output_df']!="None"):
        output_df['DICE'] = pd.Series(dice)
        
        str1 = ["Pred","Actual"]
        for i in range(len(str1)):
           output_df[str1[i] + "_FD"] = pd.Series(fd[:,i])
           output_df[str1[i] + "_Area"] = pd.Series(area[:,i])
        output_df.to_csv(arguments['--output_df'])


    
    

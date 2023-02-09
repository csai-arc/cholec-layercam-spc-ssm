import os
import sys
import time
import cv2
import numpy as np
import xlsxwriter
import pandas
import PIL.Image, PIL.ImageOps
import random
from shutil import copyfile

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import models.classification as customized_models
from cfg import parser
from sklearn.metrics import confusion_matrix,classification_report
import uuid
import fnmatch
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
import gc

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    """


    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cfg.OUT_EVAL_MODE1_PTH+'Class_confusion_matrix_mode'+str(args.mode)+'.png')

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

print("===================================================================")
print("Surgcial Phase Classification")
print("===================================================================")
use_cuda = torch.cuda.is_available()
manualSeed = random.randint(1, 10000)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)
device = torch.device('cuda')
####### Load configuration arguments
# ---------------------------------------------------------------
args = parser.parse_args()
cfg = parser.load_config(args)


if (args.mode !=1 and args.mode != 2):
    print("Input correct execution mode as --mode 1 or --mode 2")
    sys.exit(0)

if (args.eval_mode !=0 and args.eval_mode != 1):
    print("Input correct evaluation execution mode as --eval_mode 0 or --eval_mode 1")
    sys.exit(0)

#Output file to store the result
if (args.mode == 1 and args.eval_mode ==0):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE1
    outpath   = cfg.OUT_MODE1_RES_PTH
elif (args.mode == 2 and args.eval_mode ==0):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE2
    outpath   = cfg.OUT_MODE2_RES_PTH
elif (args.mode == 1 and args.eval_mode ==1):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE1
    outpath   = cfg.OUT_EVAL_MODE1_RES_PTH
elif (args.mode == 2 and args.eval_mode ==1):
    model_pth = cfg.TRAIN.MODEL_PATH_MODE2
    outpath   = cfg.OUT_EVAL_MODE2_RES_PTH
else:
    print("Incorrect execution mode or evaluation execution mode")
    sys.exit(0)

#Network architecture validation
if cfg.TRAIN.MODEL_NAME.startswith('effnetv2'):
    model = models.__dict__[cfg.TRAIN.MODEL_NAME](
                num_classes=cfg.MODEL.NUM_CLASSES, width_mult=1.,
            )
else:
    print("Invalid network architecture")
    sys.exit(0)

model=model.cuda()
if args.mode == 2:
    model = torch.nn.DataParallel(model, device_ids=None)

#Load model
print("===================================================================")
print('loading checkpoint {}'.format(model_pth))
checkpoint = torch.load(model_pth, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
best_acc = checkpoint['best_acc']
print("accuracy =", best_acc)
print("===================================================================")

#Parameters
num_classes = cfg.MODEL.NUM_CLASSES
class_names = cfg.CLASS_NAMES
input_directory = cfg.INPUT_PATH
if args.eval_mode == 1:
    input_directory = cfg.INPUT_EVAL_PATH
    for subdir in sorted(os.listdir(input_directory)):
        if os.path.isdir(os.path.join(input_directory, subdir)) is False:
            continue
        if subdir not in cfg.CLASS_NAMES:
            print("\nPlease ensure to match evaluation data classes with class names listed in configuration file.")
            sys.exit(0)

#Put model in evaluation mode
model.eval()

target_layers = [model.conv, model.features[-5], model.features[-15], model.features[-25]]
targets = None
#model.eval()
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((cfg.DATA.IMG_HEIGHT,cfg.DATA.IMG_WIDTH)),transforms.Normalize(cfg.DATA.MEAN, cfg.DATA.STD)])
transf = transforms.ToPILImage()

List_predictions_class=[]
List_gt_class=[]
List_filename=[]
List_filename0=[]
List_filename1=[]

if args.mode == 1:
    #Create excel workbook
    #print(outpath)
    Workbook = xlsxwriter.Workbook(outpath)
    for subdir in sorted(os.listdir(input_directory)):
        if os.path.isdir(os.path.join(input_directory, subdir)) is False:
            continue
        # Configure excel to log predictions
        Sheet = Workbook.add_worksheet(subdir)
        Sheet.write("A1", "IMAGE_NAME")
        Sheet.write("B1", "Srugical_Phase(Top-1)")
        Sheet.write("C1", "Score(Top-1)")
        Sheet.write("D1", "Srugical_Phase(Top-2)")
        Sheet.write("E1", "Score(Top-2)")
        Sheet.write("F1", "Srugical_Phase(Top-3)")
        Sheet.write("G1", "Score(Top-3)")
        row = 1
        col = 0
        filelist = sorted(fnmatch.filter(os.listdir(os.path.join(input_directory, subdir)),'*.jpg'))

        for filename in filelist:
            #print("Processing image: ", subdir+"/"+filename)
            scores = []
            start_time = time.time()

            img = PIL.Image.open(os.path.join(input_directory, subdir, filename))

            input_img = transform(img).float()
            input_img = torch.unsqueeze(input_img, 0).cuda()

            # Classification using trained model
            with torch.no_grad():
                outputs_class, outputs_hm = model(input_img)

            probs = torch.nn.functional.softmax(outputs_class, dim=1)[0]

            #Get Top-3 predictions
            for i in range (num_classes):
                scores.append(probs[i].item())

            a = sorted(zip(scores, class_names), reverse=True)[:3]
            if (cfg.GRAD_CAM_FLAG==1):
                   
               with LayerCAM(model=model, target_layers=target_layers, use_cuda=use_cuda) as cam:
                  cam.batch_size = 32
                
                  grayscale_cam = cam(input_tensor=input_img, targets=targets)
                  grayscale_cam = grayscale_cam[0, :]
                  dim = (cfg.DATA.IMG_WIDTH, cfg.DATA.IMG_HEIGHT)
                  resized = img.resize(dim)
                  outputs_hm = transf(outputs_hm.squeeze())
                  outputs_hm_resized = outputs_hm.resize(dim, PIL.Image.BILINEAR)
                  outputs_hm_resized1 = np.float32(np.array(outputs_hm_resized)) 
                  rgb_img = np.float32(np.array(resized)) / 255
                  cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                  cam_image = PIL.Image.fromarray(cam_image)
                  outputs_hm_resized1=outputs_hm_resized1-np.min(outputs_hm_resized1)
                  outputs_hm_resized1=outputs_hm_resized1/np.max(outputs_hm_resized1)
                  outputs_hm_resized1 = (1 - outputs_hm_resized1)
                  outputs_cam_image = show_cam_on_image(rgb_img, outputs_hm_resized1, use_rgb=False)
                  outputs_cam_image = PIL.Image.fromarray(outputs_cam_image)
                  im_h = PIL.Image.new('RGB', (resized.width + cam_image.width , resized.height))
                  im_h.paste(resized, (0, 0))
                  im_h.paste(cam_image, (resized.width, 0))
                  output_image_path = cfg.OUT_EVAL_MODE1_PTH+"LayerCAM/"
                  if not os.path.exists(output_image_path):
                     os.makedirs(output_image_path)
                  im_h.save(output_image_path+filename.split('.')[0]+"_layercam_"+subdir+'.jpg')
            
            
            end_time = time.time()

            #Log predictions/output to excel
            Sheet.write(row, col, filename)
            for i in range(0,len(a)):
                Sheet.write(row, col+1, a[i][1])
                Sheet.write(row, col+2, a[i][0])
                col = col + 2
            col = 0
            row = row + 1
            #Update ground truth and prediction class list for each image
            if args.eval_mode == 1:
                List_gt_class.append(int(subdir))
                List_predictions_class.append(int(a[0][1]))
                List_filename.append(filename)

    Workbook.close()



#Evaluation mode
if args.eval_mode == 1:
    #Output path to generate evaluation report and classwise predictions
    output_folder = cfg.OUT_EVAL_MODE1_PTH+"classwise_predictions_mode"+str(args.mode)
    path = cfg.OUT_EVAL_MODE1_PTH+"Class_eval_report_mode"+str(args.mode)+".xlsx"

    target_list=cfg.CLASS_NAMES
    labels= list(map(int, cfg.CLASS_NAMES))

    #To create classification report
    class_report=classification_report(List_gt_class, List_predictions_class,labels=labels,target_names=target_list,digits=5, output_dict=True)
    class_report_df = pandas.DataFrame(class_report).transpose()

    #To create confusion matrix
    Confusion_matrix=confusion_matrix(List_gt_class, List_predictions_class,labels=labels)
    Confusion_matrix_df = pandas.DataFrame(Confusion_matrix,index=target_list,columns =target_list)

    with pandas.ExcelWriter(path) as eval_writer:
         class_report_df.to_excel(eval_writer,sheet_name="Classification_Report")
         eval_writer_book = eval_writer.book
         cell_format1 = eval_writer_book.add_format({'bold': True, 'bg_color':'#d3e6d5'})
         eval_writer.sheets["Classification_Report"].conditional_format('C18:C18',{'type': 'cell', 'criteria': '>=', 'value':    0, 'format': cell_format1})
         eval_writer.sheets["Classification_Report"].set_column('A:A',20)
         Confusion_matrix_df.to_excel(eval_writer,sheet_name="Confusion_Matrix")

    plot_confusion_matrix(Confusion_matrix,normalize = True, target_names = target_list, title = "Confusion Matrix")

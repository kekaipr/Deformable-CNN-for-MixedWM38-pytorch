import numpy as np
import matplotlib.pyplot as plt
import time, os
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.utils import model_zoo
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from layers_train_pytorch import ConvOffset2D
import torch.nn.functional as F
import Performance_Matrices as PM
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
################
# get dataset
################
class MultiLabelWebDataset(Dataset):
  def __init__(self, root_dir, classes, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.classes = classes
    self.class_to_idx = {c:i for i, c in enumerate(self.classes)}
    self.data = self.make_dataset()                                        
                                                      
  def __len__(self):
    return len(self.make_dataset())

  def make_dataset(self):
    instances = []
    for target_class in os.listdir(self.root_dir):      
      target_dir = os.path.join(self.root_dir, target_class)
      # split up the class names by "+" sign
      class_names = target_class.split("+") # list of length 1 or 2
      if not os.path.isdir(target_dir):
        continue
      for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
        for fname in sorted(fnames):  # for each image
          label = [0]*len(self.classes)
          path = os.path.join(self.root_dir, target_class, fname)
          # if len(class_names)==1:  # images that contain only one class
          single_cls = class_names[0]
          label[self.class_to_idx[single_cls]] = 1.
          # elif len(class_names)==2:  # images that contain two classes
          #   for cls in class_names:
          #     label[self.class_to_idx[cls]] = 1.
          item = path, label
          instances.append(item)
    return instances

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    path, target = self.data[idx]
    image = Image.open(path).convert('RGB')
    image_array = np.array(image)
    first_channel = image_array[:, :, 0]
    # mapping = {0: 0, 128: 1, 255: 2}
    # mapping = {k: v if v is not None else 0 for k, v in mapping.items()}

    # # 使用np.vectorize
    # first_channel = np.vectorize(mapping.get, otypes=[np.uint8])(first_channel)
    # first_channel = np.where(first_channel is None, 0, first_channel)
    # print(first_channel[10])
    if self.transform:
      image = self.transform(first_channel)
      # plt.imshow(first_channel)
      # plt.show
    return image, torch.tensor(target)

################
# T-SNE
################
def scale_to_01_range(x):  
  ''' scale and move the coordinates so they fit [0; 1] range '''
  value_range = (np.max(x) - np.min(x))
  starts_from_zero = x - np.min(x)
  return starts_from_zero / value_range

def plot_tsne(dataloader, encoder, plot_imgs=False, model_type='resnet50'):
    # assert model_type in ['resnet50', 'vgg16'], 'model_type must be one of "resnet50" or "vgg16"!'
    encoder = encoder.cuda().eval()

    for i, (data, target, fname) in enumerate(dataloader):
        data = data.cuda()
        with torch.no_grad():
            if model_type == "resnet50":
                outputs = encoder(data)
                outputs = torch.flatten(outputs, 1)
            elif model_type == "vgg16":
                outputs = encoder[0](data)
                outputs = encoder[1](outputs)
                outputs = torch.flatten(outputs, 1)
                outputs = encoder.classifier[0](outputs)
                outputs = encoder.classifier[1](outputs)
                outputs = encoder.classifier[2](outputs)
                outputs = encoder.classifier[3](outputs)
                outputs = encoder.classifier[4](outputs)
        outputs = outputs.cpu().numpy()
        features = outputs if i == 0 else np.concatenate((features, outputs), axis=0)
        labels = target if i == 0 else np.concatenate((labels, target), axis=0)
        fnames = list(fname) if i == 0 else fnames + list(fname)

    print("# of samples : {} \n feature-dim : {}".format(features.shape[0], features.shape[1]))
    tsne = TSNE(n_components=2, init='pca', random_state=501).fit_transform(features)

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    fig = plt.figure(figsize=(12, 8))
    tx = scale_to_01_range(tsne[:, 0])
    ty = scale_to_01_range(tsne[:, 1])

    classes = dataloader.dataset.classes
    class2idx = {c: i for i, c in enumerate(classes)}
    colors = ['#FF0000', '#FF6400', '#FFDA01', '#64FF0A', '#0BFFDF', '#0029FF', '#9B00FF', '#000000', '#5A2E00']
    colors *= 2  # Repeat twice to have 20 colors
    colors_per_class = {label: colors[i % len(colors)] for i, label in enumerate(classes)}
    # print('color',colors)
    # print('colors_per_class',colors_per_class)
    if plot_imgs:
        width, height = 4000, 3000
        max_dim = 100
        full_image = Image.new('RGBA', (width, height))
        img_paths = fnames

    # fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    for label in colors_per_class:
        indices = [i for i, l in enumerate(labels) if l == class2idx[label]]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        
        if plot_imgs:
            current_img_paths = np.take(img_paths, indices)
            for img, x, y in zip(current_img_paths, current_tx, current_ty):
                tile = Image.open(img)
                rs = max(1, tile.width / max_dim, tile.height / max_dim)
                tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
                full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))
        else:
            color = colors_per_class[label]
            # Set marker as 'o' for the first 10 classes, and '^' for the rest (triangles)
            if label.startswith('811-'):
                marker = 'o' 
                # print('current_tx(811)' ,len(current_tx)) 
                ax.scatter(current_tx, current_ty, c=color, label=label, alpha=0.5, marker=marker)
            else:
                marker = '^'
                # print('current_tx(38)' ,len(current_tx)) 
                ax.scatter(current_tx, current_ty, edgecolors='black', c=color, label=label, alpha=0.5, marker=marker)

    ax.legend(loc='best')
    plt.show()

################
# confusion matrix
################

def calculate_confusion_matrix(encoder, 
                               classifier, 
                               transform, # torchvision.transforms object
                               img_dir,   # name of directory containing images to be tested (to calculate confusion matrix)
                               classes,   # list containing class names
                               multi_label=False, # True for multi-label data, False for single-label data
                               threshold=0.3):   # detection threshold for multi-label data
  encoder.cuda().eval()
  classifier.cuda().eval()
  n_classes = len(classes)
  test_labels = []
  first_img = True
  use_cuda = True
  # initialize confusion matrix 
  cm = np.zeros((n_classes, n_classes))
  for cl in os.listdir(img_dir): # for each class
    # print('cl:',cl)
    cls_i = classes.index(cl)
    for img_path in os.listdir(os.path.join(img_dir, cl)): # for each image
      test_labels.append(cls_i)
      img = Image.open(os.path.join(img_dir, cl, img_path)).convert('RGB')     
      img = transform(img)[:1, :, :].unsqueeze(0)    
      img = img.cuda() if use_cuda else imguse_cuda100
      with torch.no_grad():
        logits = encoder(img)
        logits = torch.flatten(logits, 1)
        logits = classifier(logits).cpu().detach().numpy()
        
      if multi_label:  # Multi-label Prediction 
        logits = logits[0]
        pred = [1 if prob > threshold else 0 for i, prob in enumerate(logits)]
        # if all logits < threshold, predict the one with the largest logit
        if sum(pred)==0: 
          pred = [1 if i==np.argmax(logits) else 0 for i, prob in enumerate(logits)] 
        cm[cls_i] += pred
      else:
        preds = np.argmax(logits, axis=1)
        test_preds = preds if first_img else np.concatenate((test_preds, preds), axis=0)  
        first_img = False

  if multi_label==False:
    # using sklearn.metrics.confusion_matrix
    cm = confusion_matrix(test_labels, test_preds, labels=np.arange(n_classes))
    report = classification_report(test_labels, test_preds, target_names=classes, output_dict = True,zero_division = 0.0)
  return cm ,report

################
# make_variable
################
def make_variable(tensor, volatile=False):
    ''' function to make tensor variable '''
    # Define use_cuda as True or False based on your setup
    use_cuda = True  # or False depending on your setup
    
    if use_cuda:
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

if __name__ =="__main__":
  ################
  # set transform rule
  ################
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  transform_ = transforms.Compose([transforms.ToPILImage(),
                                  #transforms.Resize((52 ,52)),
                                  #transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()])
  transform_COM = transforms.Compose([
                                #transforms.Resize((52 ,52)),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                #transforms.Reshape(52,52,3)
                                ])

  ################
  # set source data
  ################
  src_root_dir = "38"## insert your data folder here ##
  classes=['Center','Donut','Edge-loc','Edge-ring','Loc','Near-full','Normal','Random','Scratch'] 
  train_data_multi_label = MultiLabelWebDataset(src_root_dir + '/train', classes=classes, transform = transform_)
  valid_data_multi_label = MultiLabelWebDataset(src_root_dir + '/valid', classes=classes, transform = transform_)
  test_data_multi_label = MultiLabelWebDataset(src_root_dir + '/test', classes=classes, transform = transform_)
  print("Class2idx: ", train_data_multi_label.class_to_idx)
  num_workers = 0
  batch_size = 1000
  dataloaders_multi_label = {}
  dataloaders_multi_label['train'] = DataLoader(train_data_multi_label, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  dataloaders_multi_label['valid'] = DataLoader(valid_data_multi_label, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  dataloaders_multi_label['test'] = DataLoader(test_data_multi_label, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  print('Train images :', len(train_data_multi_label), ", # of training batches:", len(dataloaders_multi_label['train']))
  print('Valid images :', len(valid_data_multi_label), ", # of valid batches:", len(dataloaders_multi_label['valid']))
  print('Test images :', len(test_data_multi_label), ", # of test batches:", len(dataloaders_multi_label['test']))


  ################
  # set target train/test data
  ################
  num_workers = 0
  batch_size = 1000

  tgt_root_dir = '811/train'
  wm811_dataset = MultiLabelWebDataset(tgt_root_dir, classes = classes, transform = transform_) # resize only
  wm811_dataloader = DataLoader(wm811_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  print("Classes: ", wm811_dataset.class_to_idx)
  print("Number of wm811_test images : {}    Number of wm811_test batches : {}  (batch size={})".format(len(wm811_dataloader.dataset), len(wm811_dataloader), batch_size))

  tgt_root_dir_valid = '811/valid'
  wm811_valid_dataset = MultiLabelWebDataset(tgt_root_dir_valid, classes = classes, transform = transform_) # resize only
  wm811_valid_dataloader = DataLoader(wm811_valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  print("Classes: ", wm811_valid_dataset.class_to_idx)
  print("Number of wm811_valid images : {}    Number of wm811_test batches : {}  (batch size={})".format(len(wm811_valid_dataset), len(wm811_valid_dataloader), batch_size))

  tgt_root_dir_test = '811/test'
  wm811_test_dataset = MultiLabelWebDataset(tgt_root_dir_test, classes = classes, transform = transform_) # resize only
  wm811_test_dataloader = DataLoader(wm811_test_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
  print("Classes: ", wm811_test_dataset.class_to_idx)
  print("Number of wm811_test images : {}    Number of wm811_test batches : {}  (batch size={})".format(len(wm811_test_dataset), len(wm811_test_dataloader), batch_size))


  ################
  # set combine data (for T-SNE)
  ################
  data_path = 'combine'
  num_workers = 0
  batch_size = 1000
  
  class ImageFolderWithPaths(datasets.ImageFolder):
    ''' dataset containing images as well as image filenames '''
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        
        # Extract the first channel from the image tensor
        image_tensor = original_tuple[0][0, :, :].unsqueeze(0)  # Assuming the tensor is of shape (52, 52, 3)
        # print('image_tensor:',image_tensor.shape)
        # Create a new tuple with the modified image tensor
        tuple_with_modified_tensor = (image_tensor, original_tuple[1], path)
        
        return tuple_with_modified_tensor
  combined_dataset = ImageFolderWithPaths(data_path, transform = transform_COM) 
  # combine_classes = combined_dataset.classes
  # combined_dataset = MultiLabelWebDataset(data_path, classes=combine_classes ,  transform = transform_) 
  print('combined_dataset:',combined_dataset)
  combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  print("Classes: ", combined_dataset.classes)
  print("Number of wm811 images : {}    Number of wm811 batches : {}  (batch size={})".format(len(combined_dataset), len(combined_dataloader), batch_size))



    
  
  #######################################################################################################################
  # Training process
  #######################################################################################################################
  def train_adda(num_epochs,  # number of epochs to train
               lr,          # learning rate
               stop_num,
               save_step,   # save model checkpoint every this number of epochs 
               encoder,        
               classifier,            
               discriminator, 
               src_data_loaders,      # dict of train/valid/test source domain dataloaders
               tgt_data_loader,       # target domain dataloader for training
               tgt_data_loader_small, # target domain dataloader for testing (smaller)
               combined_dataloader,   # source + target domain dataloader for plotting t-SNE
               src_test_dir,   # file directory containing source doamin test images
               tgt_test_dir,   # file directory containing source doamin test images
               alpha_CLS=1.,      # coefficient for class classification loss in computing the total loss for encoder
               alpha_DA=1.,       # coefficient for domain adaptation (confusion) loss in computing the total loss for encoder
               multi_label=False, # True for multi-label data, False for single-label data
               test_threshold=0.3 # output threshold for considering the object as present for multi-label data
               ):
     
    ### Define loss for class-classification ###
    criterion_cls = nn.MSELoss() if multi_label else nn.CrossEntropyLoss() # MSE loss for soft-label 

    ### Define loss for domain-classification ###
    criterion_DA  = nn.CrossEntropyLoss()

    ### Define optimizers for encoder, classifier, and discriminator ###
    optimizer_encoder       = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_classifier    = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    len_data_loader  = min(len(src_data_loaders['train']), len(tgt_data_loader)) # len(tgt_unlabeled_data_loader))
    valid_loss_min = np.Inf
    prev_save = ""

    use_cuda = True
    ### Move the models to GPU ###
    if use_cuda:
      discriminator.cuda()
      encoder.cuda()
      classifier.cuda()

    classification_losses_E, domain_confusion_losses_E, losses_D, accs_D = [], [], [], []
    train_loss_E_class, train_loss_E_domain, train_loss_D, train_acc_D, target_valid_loss = 0., 0., 0., 0., 0.
    valid_loss_class, temp = 0., 0 # only check for class-classification for validation (no domain-related tasks)

    
    early_stop = {'count' :0 , 'best_epoch':0,'ACC':0}
    ### Start Training! ###
    for epoch in range(num_epochs):
      #### 1. Plot t-SNE plot with source and target domain features together ####
      if ((epoch+1)%50 == 0) or (epoch == 0):
        plot_tsne(combined_dataloader, encoder)

      ### Start timing ###
      start = time.time()       


      ####################  2. Loop through Training batches  ####################
      for step, ((images_src, tgt_src), (images_tgt, _)) in enumerate(zip(src_data_loaders['train'], tgt_data_loader)): 
        
        ##########################################################################
        #######  2.1 Train Source Encoder & Classifier with class labels  ########
        ##########################################################################
        encoder.train()
        classifier.train()
        images_src, images_tgt = make_variable(images_src), make_variable(images_tgt)
        tgt_src = tgt_src.type(torch.FloatTensor).cuda() if multi_label else make_variable(tgt_src)
        # print('tgt_src',tgt_src)
        # print('tgt_src shape',tgt_src.shape)
        optimizer_encoder.zero_grad()
        optimizer_classifier.zero_grad()

        ### Forward only SOURCE DOMAIN images through Encoder & Classifier ###
        output = encoder(images_src)    # [batch_size, n_classes]  (target: [batch_size])
        output = torch.flatten(output, 1)
        output = classifier(output)
        # print('tgt_src',output)
        # print('tgt_src shape',output.shape)
        ### Calculate class-classification loss for Encoder and Classifier ###
        loss_CLS = criterion_cls(output, tgt_src)
        train_loss_E_class += loss_CLS.item() 
        
        ##########################################################################   
        #############  2.2 Train Discriminator with domain labels  ###############
        ##########################################################################
        discriminator.train()
        optimizer_discriminator.zero_grad()

        ### Forward pass through Encoder ###
        feat_src = encoder(images_src) 
        feat_tgt = encoder(images_tgt)
        
        ### Concatenate source domain and target domain features ###
        feat_concat = torch.cat((feat_src, feat_tgt), 0) # [batch_size*2, 512, 1, 1]
        feat_concat = feat_concat.squeeze(-1).squeeze(-1)  # [batch_size*2, 512]
        # feat_concat = torch.flatten(feat_concat, 1)
        ### Forward concatenated features through Discriminator ###
        pred_concat = discriminator(feat_concat.detach())

        ### prepare source domain labels (1) and target domain labels (0) ###
        label_src = make_variable(torch.ones(feat_src.size(0)).long()) 
        label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
        label_concat = torch.cat((label_src, label_tgt), 0)

        ### Calculate domain-classification loss for Discriminator ###
        loss_discriminator = criterion_DA(pred_concat.squeeze(1), label_concat)

        # ### Backward Propagation for Discriminator ###
        loss_discriminator.backward()
        optimizer_discriminator.step()

        ### Update running losses/accuracies ###
        train_loss_D += loss_discriminator.item()
        
        pred_cls = torch.squeeze(pred_concat.max(1)[1])
        
        train_acc_D += (pred_cls == label_concat).float().mean()


        ##########################################################################
        ############  2.3 Train Source Encoder w/ FAKE domain label  #############
        ##########################################################################
        ### Forward only TARGET DOMAIN images through Encoder ###
        feat_tgt = encoder(images_tgt)

        ### Forward only TARGET DOMAIN features through Discriminator ###
        pred_tgt = discriminator(feat_tgt.squeeze(-1).squeeze(-1))     
        label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long()) # prepare fake labels
        
        ### Calculate FAKE domain-classification loss for Encoder ###
        loss_DA = criterion_DA(pred_tgt.squeeze(1), label_tgt)
        train_loss_E_domain += loss_DA.item()

        ### For encoder and Classifier, 
        ### optimize class-classification & fake domain-classification losses together ###
        loss_total = (alpha_CLS * loss_CLS) +  (alpha_DA * loss_DA)
        loss_total.backward()
        optimizer_encoder.step()
        optimizer_classifier.step()
        
        
        
      ################### target valid (for early stop) #########################

      #################### 3. Loop through Validation batches ####################
      if ((epoch+1)%1 == 0) or (epoch == 0):
        encoder.eval()
        classifier.eval()
        for data, target in src_data_loaders['valid']:
          data = make_variable(data)
          target = target.type(torch.FloatTensor).cuda() if multi_label else make_variable(target)
          with torch.no_grad():
            output = encoder(data)    # [batch_size, n_classes]  (target: [batch_size])
            output = torch.flatten(output, 1)
            output = classifier(output)

          loss = criterion_cls(output, target)
          valid_loss_class += loss.item()
          if multi_label==False:
            output = output.cpu().detach().numpy()
            # print('output:', output)
            # print('output shape:', output.shape)
            # val_n_corr_class += int(sum([np.argmax(pred)==target[i] for i, pred in enumerate(output)]))


      ####################  4. Log train/validation losses  ######################
      
      train_acc_D = train_acc_D/min(len(src_data_loaders['train']), len(tgt_data_loader))
      print('\n-----Epoch: %d/%d-----'%(epoch+1, num_epochs),'the best acc is',early_stop['ACC'],'in epoch',early_stop['best_epoch'])
      if ((epoch+1)%50 == 0) or (epoch == 0):
        print('Train Classification Loss (E,C): %.3f  Train Domain Confusion Loss (E): %.3f  Valid Classification Loss (E,C): %.3f'%(train_loss_E_class/min(len(src_data_loaders['train']), len(tgt_data_loader)), train_loss_E_domain/min(len(src_data_loaders['train']), len(tgt_data_loader)), valid_loss_class/len(src_data_loaders['valid'])))
        print('Domain Classification Loss (D): %.3f  Domain Classification Accuracy (D): %.3f  elapsed time: %.1fs'%(train_loss_D/min(len(src_data_loaders['train']), len(tgt_data_loader)), train_acc_D, time.time()-start))
        print('Total loss : ', loss_total.item())
      # if multi_label==False:
      #   valid_acc = val_n_corr_class/len(src_data_loaders['valid'].dataset)

      ### Reset running losses/accuracies to zero ###
      classification_losses_E.append(train_loss_E_class)
      domain_confusion_losses_E.append(train_loss_E_domain)
      losses_D.append(train_loss_D)
      accs_D.append(train_acc_D)
      train_loss_E_class, train_loss_E_domain, train_loss_D, valid_loss_class , train_acc_D= 0., 0., 0., 0., 0. 

      #########  5. Show confusion matrices for both domains' test sets  #########
      # set threshold=0.5 for source domain confusion matrix 
      if ((epoch+1)%50 == 0) or (epoch == 0):
        print('class:',src_data_loaders['train'].dataset.classes)
      
      cm, _ = calculate_confusion_matrix(encoder, classifier, transform=transform_COM, classes=src_data_loaders['train'].dataset.classes, 
                                      img_dir=src_test_dir, threshold=0.5, multi_label=False)#, test=True)
      
      
      if ((epoch+1)%50 == 0) or (epoch == 0):
        print("--Source Domain Confusion Matrix--")
        print(cm)
      # to be more lenient for target domain class deteciton, set threshold to be lower than 0.5 (e.g. 0.2)
      cm, report = calculate_confusion_matrix(encoder, classifier, transform=transform_COM, classes=tgt_data_loader_small.dataset.classes, 
                                      img_dir=tgt_test_dir, threshold=test_threshold, multi_label=False)#, test=True)
      
      
      if ((epoch+1)%50 == 0) or (epoch == 0):
        print("--Target Domain Confusion Matrix--")
        print(cm)
      accThisEpoch = report['accuracy']

      ######################  6. Save model checkpoints  #########################
      ### Save model if validtion loss is smaller than previous epoch's ###
      # if valid_loss_class < valid_loss_min:
      #   ### Delete previously saved model checkpoint ###
      #   if prev_save:
      #     os.remove("0119_ADDA_weight/encoder" + prev_save + ".pt")
      #     os.remove("0119_ADDA_weight/classifier" + prev_save + ".pt")
      #   prev_save = "_" + str(epoch+1) 

      #   ### Save the new (best) model checkpoints ###
      #   torch.save(encoder.state_dict(), "0119_ADDA_weight/encoder" + prev_save + ".pt")
      #   torch.save(classifier.state_dict(), "0119_ADDA_weight/classifier" + prev_save + ".pt")
      #   valid_loss_min = valid_loss_class
      ### Regularly save model checkpoints every [save_step] epochs ###
      # if ((epoch + 1) % save_step == 0):
        


      with torch.no_grad():
        for im ,target in wm811_valid_dataloader:
            im,target = im.to(device), target.to(device)
            output = encoder(im)
            output = torch.flatten(output, 1)
            output = classifier(output)
            # print('output:' ,output)
            target_valid_loss += criterion_cls(output, target)


        if (accThisEpoch)> early_stop['ACC']:
            early_stop['best_epoch'] = epoch+1
            best_epoch = early_stop["best_epoch"]
            early_stop['ACC'] = accThisEpoch
            print(f'{epoch+1}: the new accuracy is {early_stop["ACC"]}')
            early_stop['count'] = 0
            torch.save(encoder.state_dict(), "0119_ADDA_weight/ADDA-encoder-{}.pt".format(epoch + 1))
            torch.save(classifier.state_dict(), "0119_ADDA_weight/ADDA-classifier-{}.pt".format(epoch + 1))
        else:
            early_stop['count'] += 1
            
        if early_stop['count']==stop_num:
            print(f'the best epoch is {early_stop["best_epoch"]}')
            
            break
      target_valid_loss = 0.      
    return encoder, classifier, classification_losses_E, domain_confusion_losses_E, losses_D, accs_D, best_epoch
  

  ################
  # set Discriminator
  ################
  class Discriminator(nn.Module):
    def __init__(self, feature_dim):
      super(Discriminator, self).__init__()
      self.restored = False
      self.layer = nn.Sequential(
        nn.Linear(feature_dim,256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.LogSoftmax(dim=1)
      )       
    def forward(self, input):
      out = self.layer(input)
      return out

  #######################################################################################################################
  # model and encoder 
  #######################################################################################################################
  #######
  #deformable
  ######
  class Mymodel(nn.Module):
    def __init__(self, in_channels, out_class_dim, trainable=True):
        super().__init__()
        self.conv_num = 32  # Assuming this value

        self.conv_block_1= self.ConvBolck(in_channels, self.conv_num)
        self.conv_block_2= self.ConvBolck(self.conv_num, self.conv_num* 2)
        self.conv_block_3= self.ConvBolck(self.conv_num* 2, self.conv_num* 4)
        self.conv_block_4= self.ConvBolck(self.conv_num* 4, self.conv_num* 8)
        self.conv_block_5= self.ConvBolck(self.conv_num* 8, self.conv_num* 4)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.conv_num*4, out_class_dim)
        self.sigmoid = nn.Sigmoid()
    
    def ConvBolck(self, in_channels, out_channels):
        conv_block= nn.Sequential(ConvOffset2D(in_channels),
        nn.Conv2d(in_channels, out_channels, (3,3), stride=(2, 2), padding= 1),
                                                    nn.BatchNorm2d(out_channels),
                                                    nn.ReLU())
        return conv_block
#     # Deformable forward
    def forward(self, x):      
        x_1= self.conv_block_1(x)
        x_2= self.conv_block_2(x_1)
        x_3= self.conv_block_3(x_2)
        x_4= self.conv_block_4(x_3)
        x_5= self.conv_block_5(x_4)
        x_6 = self.global_avg_pool(x_5)
        # x_7 = torch.flatten(x_6, 1)
        x_7 = self.flatten(x_6)
        x_8 = self.fc(x_7)
        output = self.sigmoid(x_8)
        return output

  def define_models(n_classes):#, pretrained_on="imagenet"):
    model = Mymodel(in_channels=1,out_class_dim=9)

    classifier = nn.Sequential(*[ model.fc, model.sigmoid])
        # Define discriminator
    encoder = nn.Sequential(*[model.conv_block_1,model.conv_block_2, model.conv_block_3,model.conv_block_4,model.conv_block_5, model.global_avg_pool])#,model.flatten
    # encoder = nn.Sequential(*[model.features,model.avgpool])
    discriminator = Discriminator(feature_dim=128)
    print('encoder:',encoder)
    print('Classifier:',classifier)
    print("discriminator",discriminator)
    return encoder, classifier, discriminator
    
  
  n_classes = len(dataloaders_multi_label['train'].dataset.classes)
  encoder, classifier, discriminator = define_models(n_classes)

  print("classes : {}".format(train_data_multi_label.classes))
  encoder, classifier, classification_losses_E, domain_confusion_losses_E, losses_D, accs_D, best_epoch = train_adda(num_epochs = 100000,
                                                                                                       lr = 5e-5,
                                                                                                       stop_num = 300,
                                                                                                       save_step =1,
                                                                                                       encoder = encoder, 
                                                                                                       classifier = classifier,
                                                                                                       discriminator = discriminator,
                                                                                                       src_data_loaders = dataloaders_multi_label, 
                                                                                                       tgt_data_loader = wm811_dataloader,
                                                                                                       tgt_data_loader_small = wm811_test_dataloader,
                                                                                                       src_test_dir = "38/test", 
                                                                                                       tgt_test_dir = "811/test",
                                                                                                       combined_dataloader = combined_dataloader,
                                                                                                       alpha_CLS = 2,
                                                                                                       alpha_DA =1,
                                                                                                       multi_label = False,
                                                                                                       test_threshold=0.2)

  targetls, predls = [], []
  defect_dict = {"Center": 0,
                "Donut":  1,
                "Edge-loc": 2,
                "Edge-ring": 3,
                "Loc":4,
                "Near-full":5,
                "Normal": 6,
                "Random":7,
                "Scratch":8}

  defect_dict_inverse = {}
  for i in range(0, len(defect_dict)):
      key_name= list(defect_dict.keys())[i]
      defect_dict_inverse[str(np.array(defect_dict[key_name]))]= key_name

  classifier.load_state_dict(torch.load(f'C:/Users/MA201-Ultima/Desktop/thesis/0119_ADDA_weight/ADDA-classifier-{best_epoch}.pt')) #{best_epoch}
  encoder.load_state_dict(torch.load(f'C:/Users/MA201-Ultima/Desktop/thesis/0119_ADDA_weight/ADDA-encoder-{best_epoch}.pt'))
  classifier.to(device)
  classifier.eval()
  encoder.to(device)
  encoder.eval()

  with torch.no_grad():

    for im ,target in wm811_test_dataloader:
      # print(target.shape)
      im,target = im.to(device), target.to(device)
      output = encoder(im)
      output = torch.flatten(output, 1)
      output = classifier(output)
      result = str(np.argmax(output[0].cpu().numpy()))
      target = str(np.argmax(target[0].cpu().numpy()))

      if result in defect_dict_inverse.keys():
          predls.append(defect_dict_inverse[str(result)])
      else : 
          predls.append('Nonetype')
      targetls.append(defect_dict_inverse[str(target)])
      
  result2 = PM.confussion_maxtrix(targetls,predls,classes)
  print(result2)
  result3 = PM.measurements(result2,classes)
  print(result3)
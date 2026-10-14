"""
To train the disentangled auto encoder with any dataset
Divided the weights of the loss values of Beta-VAE as they overweigh the loss of classifier
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import time
import argparse,sys

#sys.path.insert(0,'..')
#from disent_auto_encoder_github import DE_AutoEncoder
#import OneShot_Dataset_new as tg
from models.disent_auto_encoder import DE_AutoEncoder
from datasets import OneShot_Dataset as tg

#from invert import * #Class to invert an image
from torchvision.transforms import functional as TF #TF.invert inverts a PIL image (replaces the earlier custom Invert class)




def loss_BCE_KLD(recon_x, x, mu_con,logvar_con,mu_sty, logvar_sty,lambda_val,beta_val_sty,beta_val_cont,recon_x_from_style= None):
    batch_size = x.size(0)
    assert batch_size != 0
    
    #dividing the Beta-VAE loss terms by lambda_val as its loss is much higher than classification loss 
    #logits will inlude both bce and sigmoid layer together(better than using them seperately)
    #Here reduction='sum',sums up the losses instead of taking average.As each sample has a set of losses(for each pixel) taking average directly may give a arong value. Hence sum the values and then take the average by dividing it by batch_size.
    BCE = ( nn.functional.binary_cross_entropy_with_logits(recon_x, x, 
    reduction='sum').div(batch_size)) / lambda_val
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD (KL Divergence) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # In Beta-VAE this term is multplied by the hyperparameter beta
    #KL Divergence Loss for style distribution
    KLD_sty = beta_val_sty*(-0.5 * torch.mean(1 + logvar_sty - mu_sty.pow(2) - logvar_sty.exp()))/lambda_val
    #KL Divergence Loss for content distribution
    KLD_con = beta_val_cont*(-0.5 * torch.mean(1 + logvar_con - mu_con.pow(2) - logvar_con.exp()))/lambda_val
    
    return BCE + KLD_sty + KLD_con , BCE, KLD_sty,KLD_con
    


def train_autoencoder(dataset,learning_rate,batch_size=50,num_epochs=20):
    # Load data
    print("Initializing Dataloaders...")
      
    #data loader fo loading both train and val set
    dataloaders_dict = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}
    #dataloaders_dict = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True) for x in ['train','val']}
    
    criterion = nn.CrossEntropyLoss()  
    
    optimizer = torch.optim.Adam(imgautoencoder.parameters(), lr=learning_rate, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(imgautoencoder.parameters(), lr=learning_rate, weight_decay=1e-8)
    
    #To add a step scheduler to modify the learning rate after step_size epocs by gamma(multiplicative factor)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

    train_loss_history=[]
    val_loss_history = []
    min_loss=100000.0
    max_acc =0.0
    max_acc_epoch = 0
    phase='train'
    
    print("batch_size",batch_size)
    print("length of dataloader:train",len(dataloaders_dict['train'])*batch_size)
    print("length of dataloader:val",len(dataloaders_dict['val'])*batch_size)
    print("lambda-_val used for AE loss:",(1/lambda_val))
    # Training loop
    
    start_time = time.time()
    epoch_time_history=[]
    #with torch.autograd.profiler.profile(use_cuda=True) as p: 
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print("Epoch %d" % epoch)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                #print("training phase")
                imgautoencoder.train() # Set model to training mode
            else:
                imgautoencoder.eval()   # Set model to evaluate mode
          
            for i, (images, labels) in enumerate(dataloaders_dict[phase]):
                images = images.to(device)

                labels = labels.to(device) 
                #out_x:reconstructed input,out_y:class label cosine sim value  
                out_x, out_y , mu_content,logvar_content,mu_style,logvar_style= imgautoencoder(Variable(images))
                
                optimizer.zero_grad()

                #Loss function of auto encoder consist of 4 parts
                #1. BCE between the output of the auto encoder and the input
                #2. KL Divergence between Style distribution and the prior
                #3. KL Divergence between Content distribution and the prior
                #4. Cross entropy loss between the label and cos sim values of feature vectors and weight 
                #Lambda_val is used to adjust the weight of the terms of Beta-VAE loss


                with torch.set_grad_enabled(phase == 'train'):
                   
                    loss_classifier= criterion(out_y, labels)
                    _, preds = torch.max(out_y, 1)
                    loss_AE,loss_BCE,loss_kld_s,loss_kld_c = loss_BCE_KLD(out_x,images,mu_content,logvar_content,mu_style,logvar_style,lambda_val,beta_val_sty,beta_val_cont)

                    loss= loss_classifier +  loss_AE
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                if(i%1000 == 0):#Printing the loss after 1000 batches
                    print("batch num:",i)
                    print('epoch [{}/{}], loss:{:.5f},cls loss:{:.5f},AE loss:{:.5f},BCE loss:{:.5f},KLD_s loss:{:.5f},KLD_c loss:{:.5f}'.format(epoch+1, num_epochs, loss,loss_classifier,loss_AE,loss_BCE,loss_kld_s,loss_kld_c ))
                    
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('epoch [{}/{}] Epoch Loss({}): {:.4f} ,Epoch accuracy :{:.4f},correct/total:{}/{}'.format(epoch+1, num_epochs,phase,epoch_loss,epoch_acc,running_corrects,len(dataloaders_dict[phase].dataset)))

            if(phase=='val'): 
                val_loss_history.append(epoch_loss)
                if(epoch_loss < min_loss):
                    min_loss = epoch_loss
                print("Current Min loss:",min_loss)
                if(epoch_acc > max_acc):
                    max_acc = epoch_acc
                    torch.save(imgautoencoder.state_dict(), './models_new/'+dataset_name+'_disent_ae_with_bce_with_logits_'+str(beta_val_sty)+'_con'+str(beta_val_cont)+'_lr'+str(learning_rate)+'_lam_'+str(lambda_val)+'_maxacc.pth') 
                    max_acc_epoch = epoch+1
                #elif(epoch_acc == max_acc):
                    #torch.save(imgautoencoder.state_dict(), './models_new/'+dataset_name+'_unique_class'+'_disent_ae_corrected_with_bce_with_logits_two_clslayer_wo_bias_betasty_'+str(beta_val_sty)+'_con'+str(beta_val_cont)+'_with_relu_maxpool_4bn_lr'+str(learning_rate)+'_lam_'+str(lambda_val)+'_cs_'+str(code_size[0])+'_'+str(code_size[1])+'_step_'+ str(step_size) +'_'+str(epoch+1)+'ep.pth')
                print('current max accuracy{} in {}'.format(max_acc,max_acc_epoch))

            else:
                  train_loss_history.append(epoch_loss)
            
            if(((epoch+1)%5) == 0):
                torch.save(imgautoencoder.state_dict(), './models_new/'+dataset_name+'_disent_ae_with_bce_with_logits_'+str(beta_val_sty)+'_con'+str(beta_val_cont)+'_lr'+str(learning_rate)+'_lam_'+str(lambda_val)+'_'+str(epoch+1)+'ep.pth')

            #print(p.key_averages().table(sort_by="self_cpu_time_total"))
        # To adjust the learning rate by scheduler
        scheduler.step()
        epoch_end_time = time.time()
        epoch_time_taken = epoch_end_time - epoch_start_time
        epoch_time_history.append(epoch_time_taken)
        print('Time taken for epoch [{}/{}]: {:2.5f} seconds'.format(epoch+1, num_epochs, epoch_time_taken))

    end_time = time.time()
    time_taken=end_time-start_time
    
    avg_epoch_time = sum(epoch_time_history)/len(epoch_time_history)
    num_train_samples = len(dataloaders_dict['train'].dataset)
    throughput = num_train_samples/avg_epoch_time #train images processed per second(averaged over epochs)
    print("Average time per epoch: {:2.5f} seconds".format(avg_epoch_time))
    print("Training throughput: {:2.2f} images/sec".format(throughput))
   
    if torch.cuda.is_available():
        peak_mem_allocated = torch.cuda.max_memory_allocated()/(1024**2) #in MB
        peak_mem_reserved = torch.cuda.max_memory_reserved()/(1024**2) #in MB
        print("Peak GPU memory allocated during training: {:2.2f} MB".format(peak_mem_allocated))
        print("Peak GPU memory reserved during training: {:2.2f} MB".format(peak_mem_reserved))

    print("Min loss:",min_loss)
    
    

if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description='Training with Omniglot datase')
    parser.add_argument('-df','--data_path', help='data path',required = True,  type=str)
    parser.add_argument('-nc','--num_classes', help='Number of classes', default = 1200, type=int)
    #parser.add_argument('-nc','--num_classes', help='Number of classes', default = 3856, type=int)
    parser.add_argument('-cs','--code_size', nargs='+', type=int, default=[1000,500])
    parser.add_argument('-is','--input_size',help='dimension of the input image',default=28,type=int)     
    parser.add_argument('-nic','--no_of_in_channels',nargs='+', type=int, default=[1,128,256,512])  
    parser.add_argument('-noc','--no_of_out_channels', nargs='+', type=int, default=[128,256,512,1024])
    parser.add_argument('-ne','--num_epochs', help='Number of epochs', default = 40, type=int)
    parser.add_argument('-lr','--learning_rate', help='Learning rate', default = 0.00002, type=float)
    parser.add_argument('-bs','--batch_size', help='batch size', default = 10, type=int)
    parser.add_argument('-sdf','--state_dict_file', help='state dict file', default = "none", type=str)
    parser.add_argument('-ds','--dataset', help='dataset', default = "omniglot", type=str)
    parser.add_argument('-bvs','--beta_val_style',help='beta value for style',default=1,type=float)
    parser.add_argument('-bvc','--beta_val_content',help='betavalue for content',default=1,type=float)
    parser.add_argument('-lv','--lambda_val',help='Lambda value for KLD loss',default=1,type=float)
    parser.add_argument('-ss','--step_size',help='step size (epochs) of the learning-rate scheduler; default: num_epochs (no decay)', default=None, type=int)       


    args = vars(parser.parse_args())
    is_all=is_testclass=False
    datafilepath=args['data_path']
    code_size=args['code_size']
    input_size=args['input_size']
    num_classes=args['num_classes']
    num_epochs=args['num_epochs']
    batch_size=args['batch_size']
    learning_rate=args['learning_rate']
    state_dict_file=args['state_dict_file']
    no_of_in_channels=args['no_of_in_channels']
    no_of_out_channels=args['no_of_out_channels']
    dataset_name = args['dataset']
    beta_val_sty = args['beta_val_style']
    beta_val_cont = args['beta_val_content']
    lambda_val = args['lambda_val']
    if(args['step_size'] is None):
        step_size = num_epochs
    else:
        step_size = args['step_size']



    if(dataset_name =='omniglot'):
        dataset_class = tg.OmniglotDataset
        num_classes = 1200
        num_classes_split =1200 #Train and test classes are in the same folder hence need  the split point
        images_per_class = 20
        train_num = 18
        no_of_datafolders = 2
        input_size =28
        cls_name_folders = 2
        no_of_in_channels[0] = 1

        data_trans = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])
        data_trans_omniglot_daug_rrot =  transforms.Compose([
            transforms.Resize(input_size),
            #Invert(),
            transforms.Lambda(TF.invert),
            #transforms.RandomRotation((-20,-20)),#((-1,-5)),
            transforms.RandomRotation((-10,-2)),#((-1,-5)),
            #Invert(),
            transforms.Lambda(TF.invert),
            transforms.ToTensor()
        ])

        data_trans_omniglot_daug_lrot =  transforms.Compose([
                transforms.Resize(input_size),
                #Invert(),
                transforms.Lambda(TF.invert),
                #transforms.RandomRotation((+10,+10)),#((-1,-5)),
                transforms.RandomRotation((+2,+10)),#((-1,-5)),
                #Invert(),
                transforms.Lambda(TF.invert),
                transforms.ToTensor()
            ])

        data_trans_omniglot_daug_aff =  transforms.Compose([
                transforms.Resize(input_size),
                #Invert(),
                transforms.Lambda(TF.invert),
                #transforms.Grayscale(),
                transforms.RandomAffine(10,shear=(1.5,2.5)),
                #Invert(),
                transforms.Lambda(TF.invert),
                transforms.ToTensor()
            ])



    elif(dataset_name == 'IndicSixPrinted'):
        dataset_class = tg.IndicSixPrintedDataset
        num_classes = 800#Total 1241
        num_classes_split = 800 #Train and test classes are in the same folder hence need  the split point
        images_per_class = 100 #Though some classes have less than 100 images this is set as 100 as -1 in train_num will handle it
        train_num = -1 #if this value is -1 then the last few images are taken as validation.Used mainly to handle the case where the number of images per class varies
        val_num = -5 #no of images taken for validation set from the end of training set folders
        no_of_datafolders = 2
        input_size = 50
        cls_name_folders = 2
        no_of_in_channels[0] = 3

        data_trans = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()
           ]) 

    if(no_of_datafolders ==2): 
        train_character_folders,test_character_folders,train_class_labels_dict,test_class_labels_dict = dataset_class.get_folders(datafilepath,num_classes_split)
        #Added to set shuffle as false to take only the last few images as validation set.
        if(dataset_name in ('IndicSixPrinted','IndicSixPrintedLess')):
            if(val_num>0):#To restrict the number of images taken for validation set  and it is taken consecutively after the training images (train_num)         
                task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=-1,query_per_class=val_num,num_train_classes=num_classes,class_name_folders =cls_name_folders,shuffle=False)
            else:  #Here validation set id taken at the end of the folder (train_num= -1)         
                task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=-1,query_per_class=-1,num_train_classes=num_classes,class_name_folders =cls_name_folders,val_num=val_num,shuffle=False)
        else:
            task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=-1,query_per_class=-1,num_train_classes=num_classes,class_name_folders =cls_name_folders,shuffle=False)
    else:  
        print("creating char folders")
        train_character_folders, val_character_folders,test_character_folders,train_class_labels_dict,val_class_labels_dict,test_class_labels_dict = dataset_class.get_folders(datafilepath,num_classes_split)    
        task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=-1,query_per_class=-1,num_train_classes=num_classes,class_name_folders =cls_name_folders)

    """
    print("train:",train_class_labels_dict)

    print("test", test_class_labels_dict)
    """
    is_train = True

    if(is_train):
        #Datasets with righ, left rotations and affine transform augmentations
        if(dataset_name in ('omniglot' ,'omniglot_var1')):
            dataset_plain = {x:dataset_class(task_train,split=x,transforms=data_trans) for x in ['train','val']}
            dataset_rrot = {x:dataset_class(task_train,split=x,transforms=data_trans_omniglot_daug_rrot) for x in ['train','val']}
            dataset_lrot = {x:dataset_class(task_train,split=x,transforms=data_trans_omniglot_daug_lrot) for x in ['train','val']}
            dataset_aff = {x:dataset_class(task_train,split=x,transforms=data_trans_omniglot_daug_aff) for x in ['train','val']}
        #train_autoencoder_model2(datasets,learning_rate,batch_size,num_epochs)
            datasets={x:dataset_plain[x] + dataset_rrot[x] + dataset_lrot[x] + dataset_aff[x] for x in ['train','val']} 
            #datasets={x:dataset_plain[x] for x in ['train','val']} 

        else: #For datsets without augmentation

            datasets = {x:dataset_class(task_train,split=x,transforms=data_trans) for x in ['train','val']}

    print("dataset length:",len(datasets['train']),len(datasets['val']))
    print("Step size of learning rate:",step_size)

    print(no_of_in_channels,no_of_out_channels)
    print(code_size)
    # Instantiate model
    imgautoencoder = DE_AutoEncoder(code_size,input_size,no_of_in_channels,no_of_out_channels,num_classes,is_max_pool_after_last_conv=1,is_same_style_dim=1)


    if(state_dict_file != 'none'):

     # Detect if we have a GPU available
        print("loading the weights")
        state_dict = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
        imgautoencoder.load_state_dict(state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("device:",device)
        imgautoencoder= imgautoencoder.cuda()
    print("Model:")
    print(imgautoencoder)




    os.makedirs('./models_new', exist_ok=True)
    filename='./models_new/'+ dataset_name +'_unique_class'+'_disent_ae_with_bce_with_logits_'+str(beta_val_sty)+'_con'+str(beta_val_cont)+'_lr'+str(learning_rate)+'_lam_'+str(lambda_val)+'_'+str(num_epochs)+'ep.pth'


    train_autoencoder(datasets,learning_rate,batch_size,num_epochs)
    torch.save(imgautoencoder.state_dict(), filename)

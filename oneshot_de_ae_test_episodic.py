"""
Episodic (N-way K-shot) testing of the disentangled auto-encoder (DE_AutoEncoder).

For every test episode N classes are sampled; K support images per class are encoded, the
L2-normalised mean content feature of each class becomes the weight vector of a new cosine-similarity
classifier (replacing `class_linear_2`), and the query images of those classes are classified.
The mean accuracy and its 95% confidence interval over `test_episodes` are printed for each of
`outer_episodes` repetitions, followed by the average over repetitions and the inference time.

Modes (-mc): 1 = classical FSL on the novel (test) classes only; 2 = generalized FSL, where the episode
contains both base (training) and novel classes. Mode 0 (not used in the paper) evaluates the training
classes on their validation images.
-nct is N; if not given, all test classes of the dataset are used (Omniglot 423, IndicSixPrinted 441).
-spc is K (support images per class). -qpc is the number of query images per class; if not given, all
remaining images of the class (excluding the support images) are used.
The episodes are fixed (the dataset code seeds Python’s random generator with 1), so the same checkpoint gives the same accuracies on every run.
The model must be built exactly as in training (-cs, -nic, -noc, is_max_pool_after_last_conv=1,
is_same_style_dim=1; -cs[0] is the content code size used in training); the checkpoint is given with -sdf.

Example (Omniglot, 5-way 1-shot, novel classes):
  python oneshot_de_ae_test_episodic.py -df /path/to/omniglot_resized -ds omniglot -mc 1 -cs 512 \
         -nic 1 128 256 512 -noc 128 256 512 1024 -sdf ./saved_models/<checkpoint>.pth \
         -nct 5 -spc 1 -oe 1 -te <test_episodes>
"""
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import numpy as np
import argparse
import time
import scipy as sp
import scipy.stats
from models.disent_auto_encoder import DE_AutoEncoder
from datasets import OneShot_Dataset as tg


    
#converts the dictionaries to list according to the order of the key(label) values
def dict_to_list(dict_var,ref_dict_var):
    ref_list=list(ref_dict_var.values())
    #print(ref_list)
    dict_list=[]
    for item in ref_list:
        if item in list(dict_var.keys()):#condition added to handle the case where only a subset of test classes are used
            #print("label addded:",item)
            dict_list.append(dict_var[item])
    return dict_list

#This takes the test samples randomly as we have modified the dataset class to shuffle the input
def get_mean_for_test_train_new(dataset_test_train,sample_per_class):   
    
    test_class_fc_mean_feat_dict={}
    test_class_fc_mean_feat_vect_list=[]
    
    #No of samples of test classes
    dataloader = torch.utils.data.DataLoader(dataset_test_train, batch_size=sample_per_class, shuffle=False, num_workers=4)
  
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            #print(images.shape)
            images = images.to(device)
            labels = labels.to(device) 
            out_label = labels[0].item()
            #print(out_label)
            
            out_fc  = imgautoencoder.extract_fc(images)
            
            test_class_fc_mean_feat_dict[out_label] = torch.mean(out_fc,0)
            
    #print(test_class_fc_mean_feat_dict.keys())
    test_class_fc_mean_feat_vect_list = dict_to_list(test_class_fc_mean_feat_dict,test_class_labels_dict)
    #print(len(test_class_fc_mean_feat_vect_list))
        
    
    return test_class_fc_mean_feat_vect_list


    

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    if n <= 1:
        return (float(a[0]), 0.0) if n == 1 else (0.0, 0.0)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return m,h                              


#Accumulators to store per-batch inference timing across all test episodes(computational efficiency profiling - inference time)
batch_inference_times=[]   #stores time taken(seconds) for the model forward pass, one entry per batch
batch_sizes_list=[]        #stores the actual number of images in that batch(last batch of an episode can be smaller than batch_size)

def main():
    print("started")
    #outer_episodes = 5
    tot_acc =0.0
   
    
    weight_train= imgautoencoder.class_linear_2.weight.data
    for cur_outer_ep in range(outer_episodes):
        #test_episodes=1000
        acc_list=[]
        acc_tot=0.0
        #sample_per_class=1
        #test_class_mean_feat_vect_list,test_class_fc_mean_feat_dict = get_mean_for_test_train(test_character_folders)
        test_class_mean_feat_vect_list=[]
        print('{}-Way {}-shot with {} queries'.format(num_classes_test,sample_per_class,query_per_class))
        for cur_test_ep in range(test_episodes):
            #Only novel classes are considered (Classical FSL)
            if (is_testclass):
                num_classes_from_test = num_classes_test
                #print("test class label dict:",test_class_labels_dict)
                task =tg.DatasetTask(test_character_folders,test_class_labels_dict,train_num=sample_per_class,images_per_class=images_per_class,kway=num_classes_from_test,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders)
                dataset_test_train = dataset_class(task,split='train',transforms=data_trans)
                
                test_class_mean_feat_vect_list = get_mean_for_test_train_new(dataset_test_train,sample_per_class)
                
                dataset_test = dataset_class(task,split='test',transforms=data_trans)
            
            #Only Training classes are considered 
            elif(not is_all):
                #print("in train loop")
                #To include training classes(validation dataset)
                num_classes_from_train = num_classes_test
                if(train_num!=-1):#omniglot case (Fixed number of images per class,images after the top train_num images are taken as validation set)
                    task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=num_classes_from_train,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders,shuffle=False)
                else:#IndicSixPrinted case (Variable number of images per class; last fixed num samples(val_num) taken as validation set)
                    task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=num_classes_from_train,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders,val_num=val_num,shuffle=False)
                
                dataset_val =  dataset_class(task_train,split='test',transforms=data_trans)
            
            #Both training and novelclasses are considered(Generalized FSL)
            else:
                #print("in is all loop")
                #Check whether current N in N-way K-shot is lesser than total number pof test classes
                if(num_classes_test <= tot_num_test_classes):
                    num_classes_from_test=random.randint(1,num_classes_test)
                    num_classes_from_train = num_classes_test-num_classes_from_test
                else:
                    #In this case all training and test classes are used (1220/423,800/441)
                    num_classes_from_train= num_classes#1200 in Omniglot
                    num_classes_from_test = tot_num_test_classes#num_classes_test#423
                #print('Base classes:{},novel classes:{}'.format(num_classes_from_train,num_classes_from_test))
                if(train_num!=-1):#omniglot case
                    task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=num_classes_from_train,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders,shuffle = False)
                else:#IndicSixPrinted case
                    task_train =tg.DatasetTask(train_character_folders,train_class_labels_dict,train_num=train_num,images_per_class=images_per_class,kway=num_classes_from_train,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders,val_num=val_num,shuffle=False)
                dataset_val =  dataset_class(task_train,split='test',transforms=data_trans)
                task =tg.DatasetTask(test_character_folders,test_class_labels_dict,train_num=sample_per_class,images_per_class=images_per_class,kway=num_classes_from_test,query_per_class=query_per_class,num_train_classes=num_classes,class_name_folders =cls_name_folders,con_test_labels=True)
                
                dataset_test_train = dataset_class(task,split='train',transforms=data_trans)
                test_class_mean_feat_vect_list = get_mean_for_test_train_new(dataset_test_train,sample_per_class)
                dataset_test = dataset_class(task,split='test',transforms=data_trans)
                
                dataset_all =dataset_val +dataset_test
                #print("length of test_class_mean_feat_vect_list",len(test_class_mean_feat_vect_list))
           
            if(is_all or not is_testclass):
                selected_label_list_train=[element[1] for element in dataset_val]
                selected_label_list_train = np.unique(selected_label_list_train)
                selected_label_list_train.sort()
                selected_char_label_map_train=np.arange(0,num_classes_from_train)
                #Maps the episodic label with the original trainng class label
                selected_char_dict_train={selected_char_label_map_train[i]:selected_label_list_train[i] for i in range(0,num_classes_from_train)}
                #print("dict in train:",selected_char_dict_train)
            if(is_testclass):
                selected_label_list=[element[1] for element in dataset_test]
                selected_label_list = np.unique(selected_label_list)
                selected_label_list.sort()
                #Maps the numbers 0 to numclasses_from_test-1(key) to the selected test class labels(ACCORDING TO TEST CLASS LABEL DICT)(values) by sorting them in ascending order.
                selected_char_label_map=np.arange(0,num_classes_from_test)
                selected_char_dict={selected_char_label_map[i]:selected_label_list[i] for i in range(0,num_classes_from_test)}
                #print("dict in test:",selected_char_dict)
            if(is_all):
                selected_label_list=[element[1] for element in dataset_test]
                selected_label_list = np.unique(selected_label_list)
                selected_label_list.sort()
                #print("selected_label_list",selected_label_list)
                #print("num_classes_test",num_classes_test)
                selected_char_label_map=np.arange(num_classes_from_train,num_classes_from_train+num_classes_from_test)
                selected_char_dict={selected_char_label_map[i]:selected_label_list[i] for i in range(0,num_classes_from_test)}
                selected_char_dict_train.update(selected_char_dict)
                #print("dict in all:",selected_char_dict_train)

            
            
            if (is_all or is_testclass):
                new_weight = torch.zeros(num_classes_from_test, code_size[0])
            if not is_testclass:
                new_weight_train = torch.zeros(num_classes_from_train, code_size[0])
            if(is_all or is_testclass):
                for i in range(num_classes_from_test):
                    tmp = test_class_mean_feat_vect_list[i]
                    #print(i,selected_char_dict[i])
                    #tmp = test_class_mean_feat_vect_list[selected_char_dict[i]] # to load only the selected classes when we find fc for all classes
                    new_weight[i] = tmp / tmp.norm(p=2)

                #weight = torch.cat((imgautoencoder.classifier.fc.weight.data, new_weight.cuda()))
                weight =  new_weight.to(device)#cuda()
            
            
            
            if(is_all):#validation and test dataset
                old_weight= weight_train.clone()
                for i in range(num_classes_from_train):
                    tmp = old_weight[selected_char_dict_train[i]] # to load only the selected classes when we find fc for all classes
                    new_weight_train[i] = tmp #the trained weights are used as they are (no re-normalisation)
                #weight = torch.cat((imgautoencoder.class_linear_2.weight.data, new_weight.cuda()))
                #imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_test+1200, bias=False)
                #imgautoencoder.class_linear_2.weight.data = weight
                weight = torch.cat((new_weight_train.to(device), new_weight.to(device)))
                
                #imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_test, bias=False)
                imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_from_test, bias=False)
                imgautoencoder.class_linear_2.weight.data = weight
            elif(is_testclass):#testdataset alone
                #imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_test, bias=False)
                imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_from_test, bias=False)
                imgautoencoder.class_linear_2.weight.data = weight
                
            else:
                #in case where valaidation set alone is provided no need to change the linear layer and it's weight
                old_weight= weight_train.clone()#original trained weights (class_linear_2 is replaced after every episode)
                for i in range(num_classes_from_train):
                    tmp = old_weight[selected_char_dict_train[i]] # to load only the selected classes when we find fc for all classes
                    new_weight_train[i] = tmp 
                #imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_test, bias=False)
                imgautoencoder.class_linear_2 = nn.Linear(code_size[0], num_classes_from_train, bias=False)
                imgautoencoder.class_linear_2.weight.data = new_weight_train.to(device)
            
            #To handle the case where only a subset of test classes are taken
            #test_class_mean_img_list,test_class_mean_feat_vect_list,test_class_mean_input_img_list= get_mean_for_test_train(meta_test_subfolders)
            # Load data
            #print("Initializing Dataloaders...")
            imgautoencoder.eval()    
            if(is_testclass):
                dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4) 
            elif(is_all):
                dataloader = torch.utils.data.DataLoader(dataset_all, batch_size=batch_size, shuffle=True, num_workers=4) 
            else:
                dataloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4) 
            
            # Training loop
            start_time = time.time()
            #imgautoencoder.eval()   # Set model to evaluate mode
            running_corrects = 0
            #out_fc_with_label = []
            with torch.no_grad():
                for i, (images, labels) in enumerate(dataloader):
                    #print(images.shape)
                    #print(labels)
                    images = images.to(device)

                    labels = labels.to(device) 
                    #Timing only the model forward pass(the actual inference step) - computational efficiency profiling
                    if(device.type == 'cuda'):
                        torch.cuda.synchronize()   #wait for all prior queued GPU ops so the timer starts cleanly
                    batch_infer_start = time.time()
                    #print("labels:",labels)
                    out_y , mu_content,logvar_content,mu_style,logvar_style= imgautoencoder(Variable(images)) #out_y:class label cosine sim value
                    
                    if(device.type == 'cuda'):
                        torch.cuda.synchronize()   #wait for the forward pass to actually finish on the GPU before stopping the timer
                    batch_infer_end = time.time()
                    batch_inference_times.append(batch_infer_end-batch_infer_start)
                    batch_sizes_list.append(images.size(0)) #actual batch size(handles last/partial batch correctly)


                    _, preds = torch.max(out_y, 1)
                    
                    if(is_testclass):
                        #print("predictions:",[selected_char_dict[i] for i in preds.data])
                        labels_by_classifier =[list(selected_char_dict.keys())[list(selected_char_dict.values()).index(item)] for item in labels.data]
                        labels_by_classifier = torch.tensor(labels_by_classifier).to(device)
                        #print("Target:",labels_by_classifier)
                    else:
                        labels_by_classifier =[list(selected_char_dict_train.keys())[list(selected_char_dict_train.values()).index(item)] for item in labels.data]
                        labels_by_classifier = torch.tensor(labels_by_classifier).to(device)
                        #print("labels_by_classifier:",labels_by_classifier)
                    
                    
                    running_corrects += torch.sum(preds == labels_by_classifier)

                acc = running_corrects.double() / len(dataloader.dataset)

                if((test_episodes < 50) or (test_episodes <= 100 and (cur_test_ep%10)==0)):
                    print('test episode:{},accuracy :{:.4f},correct/total:{}/{}'.format(cur_test_ep,acc,running_corrects,len(dataloader.dataset)))
                count= len(dataloader.dataset)

                

                acc_tot += acc
                acc_list.append(acc.item())
        
        #print("avg_acc:",acc_tot/test_episodes)
        #print(acc_list)
        avg_acc,h = mean_confidence_interval(acc_list)
        print("average acc:",avg_acc,h)
        tot_acc += avg_acc
    print("accuracy:",tot_acc/outer_episodes)
    #Inference time summary(computational efficiency profiling)-computed by averaging per-batch time across all batches/episodes
    if(len(batch_inference_times) > 0):
        batch_inference_times_arr = np.array(batch_inference_times)
        batch_sizes_arr = np.array(batch_sizes_list)
        avg_batch_time = batch_inference_times_arr.mean()          #average time taken per batch(seconds)
        std_batch_time = batch_inference_times_arr.std()
        #Per-image time found by averaging: total time across all batches / total number of images across all batches
        #(more accurate than avg_batch_time/avg_batch_size when the last/partial batch has a different size)
        avg_image_time = batch_inference_times_arr.sum()/batch_sizes_arr.sum()
        throughput = 1.0/avg_image_time   #images per second
        print("_"*60)
        print("Inference time summary(averaged over {} batches, {} images):".format(len(batch_inference_times),int(batch_sizes_arr.sum())))
        print("Average batch inference time: {:.6f} sec (std: {:.6f})".format(avg_batch_time,std_batch_time))
        print("Average per-image inference time: {:.6f} sec ({:.3f} ms)".format(avg_image_time,avg_image_time*1000))
        print("Inference throughput: {:.2f} images/sec".format(throughput))
        print("_"*60)
    


if __name__=="__main__":    
    parser = argparse.ArgumentParser(description='Testing disentangled vae classifier')
    parser.add_argument('-df','--data_path', help='data path',required = True,  type=str)
    parser.add_argument('-mc','--classify_mode', help='mode of classification(train-0,test-1,both-2)', default=1, choices=[0,1,2], type=int)
    parser.add_argument('-cs','--code_size', nargs='+', type=int, default=[1000,500])
    parser.add_argument('-is','--input_size',help='dimension of the input image',default=28,type=int)      
    parser.add_argument('-nic','--no_of_in_channels', nargs='+', type=int, default=[1,128,256,512])
    parser.add_argument('-noc','--no_of_out_channels', nargs='+', type=int, default=[128,256,512,1024])
    parser.add_argument('-bs','--batch_size', help='batch size', default = 10, type=int)
    parser.add_argument('-sdf','--state_dict_file',help='state dict file with path',required= True, type=str)  
    parser.add_argument('-ds','--dataset', help='dataset', default = "omniglot", type=str)
    parser.add_argument('-oe','--outer_episodes', help='number of outer iterations', default = 1, type=int)
    parser.add_argument('-te','--test_episodes', help='number of inner iterations', default = 1, type=int)
    parser.add_argument('-spc','--sample_per_class', help='how many samples to be taken from each class', default = 1, type=int)
    parser.add_argument('-qpc','--query_per_class', help='how many query images to be taken from each class', default = -1, type=int)
    parser.add_argument('-nct','--num_test_class', help='N-way', default = -1, type=int)
    
    
    args = vars(parser.parse_args())
    
    is_all=is_testclass=False
    datafilepath=args['data_path']
    code_size=args['code_size']
    input_size=args['input_size']
    no_of_in_channels=args['no_of_in_channels']
    no_of_out_channels=args['no_of_out_channels']
    batch_size = args['batch_size']
    state_dict_file=args['state_dict_file']
    if(args['classify_mode']==2):
        is_all=True   
    elif(args['classify_mode']==1):
        is_testclass=True  
    dataset_name = args['dataset']
    outer_episodes = args['outer_episodes']
    test_episodes = args['test_episodes']
    sample_per_class = args['sample_per_class']
    query_per_class= args['query_per_class'] 
    


    if(dataset_name =='omniglot'):
        dataset_class = tg.OmniglotDataset
        num_classes = 1200
        num_classes_split =1200 #Train and test classes are in the same folder hence need  the split point
        num_classes_test = 423 #423,464,20,423+41(english alph)        
        train_num = 18
        #num_classes_test = 423 #423,464,20,423+41(english alph)
        images_per_class = 20
        no_of_datafolders = 2
        no_of_in_channels[0] = 1
        #no_of_in_channels[0] = 3
        cls_name_folders = 2
        input_size =28
        data_trans = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])

    elif dataset_name == 'IndicSixPrinted':
        dataset_class = tg.IndicSixPrintedDataset
        num_classes = 800 #800 #Total 1241
        num_classes_split = 800 #800 #Train and test classes are in the same folder hence need  the split point
        num_classes_test = 441
        images_per_class = 100 #Though some classes have less than 100 images this is set as 100 as -1 in train_num will handle it
        train_num = -1 #if this value is -1 then the last few images are taken as validation.Used mainly to handle the case where the number of images per class varies
        val_num = -5 #no of images taken for validation set from training set folders
        no_of_datafolders = 2
        no_of_in_channels[0] = 3
        input_size = 50
        cls_name_folders = 2
        data_trans = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()
        ])   

    #This stores the total number of test classes which is required in case of fine tuning
    tot_num_test_classes = num_classes_test

    if(args['num_test_class']) > 0:
        num_classes_test = args['num_test_class']



    if(no_of_datafolders ==2): 
        train_character_folders,test_character_folders,train_class_labels_dict,test_class_labels_dict = dataset_class.get_folders(datafilepath,num_classes_split)
    else: 
        train_character_folders,val_character_folders,test_character_folders,train_class_labels_dict,val_class_labels_dict,test_class_labels_dict = dataset_class.get_folders(datafilepath,num_classes_split)




    print("retrieved the folders") 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device:",device)

    print(state_dict_file)
    #Move this down the summary part to calculate  the summary as state dict file is not available     
    state_dict = torch.load(state_dict_file, map_location=lambda storage, loc: storage)


    imgautoencoder = DE_AutoEncoder(code_size,input_size,no_of_in_channels,no_of_out_channels,num_classes,is_max_pool_after_last_conv=1,is_same_style_dim=1,is_train=0)


    #print(imgautoencoder)
    imgautoencoder.load_state_dict(state_dict)

    print("created the model")
    if torch.cuda.is_available():
        #print("device:",device)
        imgautoencoder=imgautoencoder.to(device)


    imgautoencoder.eval()    



    main()

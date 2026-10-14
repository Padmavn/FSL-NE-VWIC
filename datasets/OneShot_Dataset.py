# code is based on https://github.com/floodsung/LearningToCompare_FSL/blob/master/omniglot/task_generator.py
"""
Single Python file to handle all datasets.
Task generator part is commin to all datasets
This customdataset is created to assign class labels as the numeric part of the class names.
Otherwise the classlabel will be assigned randomly according to the order used by the data loader
"""
import torch
import random
from torch.utils.data import Dataset
import os,numpy as np
from PIL import Image
import pandas as pd #Added to read csv file data



# ---------- utils ----------
#To handle datsets available in csv file
def read_classes(txt_path):
    with open(txt_path) as f:
        classes = [line.strip() for line in f if line.strip() != ""]
    return list(dict.fromkeys(classes))  # preserve order, remove dups

def build_class_to_idx(classes):
    #return {str(c): i for i, c in enumerate(classes)}
    return {int(c): i for i, c in enumerate(classes)}

def load_and_filter_csv(csv_path, allowed=None):
    df = pd.read_csv(csv_path,header=None)
      
    if allowed is not None:
        df = df[df.iloc[:,0].isin(allowed)].reset_index(drop=True)
    return df

def build_gzsl_class_to_idx(seen_classes, unseen_classes):
    """
    Map seen classes  -> 0..n-1
    Map unseen classes -> n..n+m-1
    """
    #seen_class_to_idx = {str(c): i for i, c in enumerate(seen_classes)}
    seen_class_to_idx = {int(c): i for i, c in enumerate(seen_classes)}
    #This is to to return test class label dict with 0 to num_seen_classes-1 as keys to be compatible with other custom datasets
    #unseen_class_to_idx = {str(c): i for i, c in enumerate(unseen_classes)}
    unseen_class_to_idx = {int(c): i for i, c in enumerate(unseen_classes)}
    start = len(seen_classes)
    #This one has continuos labels to have a combined dict with continuos labels for seen +unseen
    #unseen_class_to_cnt_idx = {str(c): i+start for i, c in enumerate(unseen_classes)}
    unseen_class_to_cnt_idx = {int(c): i+start for i, c in enumerate(unseen_classes)}
    class_to_idx = {**seen_class_to_idx, **unseen_class_to_cnt_idx}
    return class_to_idx, seen_class_to_idx, unseen_class_to_idx

#general task generator class
class DatasetTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, either we use all 'tot_num_images_per_class' as samples without validation set (empty here) or a last few images from it is used as validation set
    # For meta testing, we use 1 or 5 shot samples for training, and use the 'query_per_class' of samples for validation.
    # If kway = -1 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If kway = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    # kway=-1 denotes that all test/train classes of the dataset(ex:miniimagenet:20) are taken for testing
    # query_per_class= -1 denotes that all remaining images other than sample images are taken for testing
    # train_num denotes the number of images taken as samples from each class. when it is -1, the images except the val_num imags from the end are used as training samples
    # num_train_classes denote the total number of training classes in the dataset
    # class_name_folders denote the number of folder names to be used in creating class name
    # set_random_seed (default: False): if True (and shuffle=True), each class's images are shuffled with a fixed seed (1) BEFORE the train/validation split, so the validation set is a reproducible random subset of the class instead of the last images in sorted filename order.Needed for datasets where the images in a class folder are ordered (e.g. by signer / session / background), so the first and last images come from different distributions and the "last images as validation set" rule gives a biased, poor validation accuracy.
    #is_val_percentage(True) and val_per is used to set the val_per percentage of validation images in the original sample set  as validation images (val_num)
    
    def __init__(self, character_folders,labels,train_num,images_per_class,con_test_labels=False,kway=-1,query_per_class=-1,num_train_classes=0,class_name_folders =1,shuffle=True,val_num=-2,set_random_seed = False, is_val_percentage=False,val_perc =0):
        samples = dict()
        self.train_roots=[]
        self.test_roots=[]
        self.train_labels=[]
        self.test_labels=[]
        #self.transforms = transforms
        self.character_folders = character_folders
        self.kway = kway
        self.query_per_class =query_per_class
        self.class_name_folders = class_name_folders
        self.images_per_class = images_per_class
        self.shuffle= shuffle
        self.set_random_seed = set_random_seed
        if(kway == -1):
            class_folders = self.character_folders 
        else:
            class_folders = random.sample(self.character_folders,self.kway)
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            temp.sort()
            #To make the validation set constant, during training only top train_num images should ne taken as train images,whereas in case of testing random image should be taken as train image.
            if(shuffle):
                # Added if condition to take the validation set by shuffling 
                if(set_random_seed == True):#Set as true when the data is ordered based on the writer style/background
                    random.seed(1)
                samples[c] = random.sample(temp, len(temp)) # contains all image files of each of the selected character folders in a random order(for all c)
            else:
                samples[c] = temp
            #Whether to fetch the last few elements(total-train_num) as validation set and the 
            #remaining as training set based on train_num variable
            if(train_num==self.images_per_class): #No validation dataset
                  # train_roots  contain the sample image filenames under the given data_root/split_fname folder
                self.train_roots += samples[c][:]
            #To handle datasets where each folder has different number of images 
            elif(train_num== -1):#in datasets where number of validation images(taken from the end) is specified rather than number of training images
                #print("inside train num=-1")
                """
                self.train_roots += samples[c][:-2]
                self.test_roots += samples[c][-2:]
                """
                #Adding is_val_percentage to use val_num as the number or percentage
                if(is_val_percentage):
                        val_num = -int(val_perc*len(samples[c])/100)
                        #print("class:{},total samples:{},val samples:{}".format(c,len(samples[c]),val_num))
                self.train_roots += samples[c][:val_num]
                if (self.query_per_class == -1):#take all images
                      self.test_roots += samples[c][val_num:]
                else:
                      self.test_roots += samples[c][val_num:][:self.query_per_class]
            else:#(Train,val) datasets in case of training,(Train,test) datasets in case of testing
                #in each class folder first train_num images are included as training images
                self.train_roots += samples[c][:train_num]
                #in each folder the last few images(total-train_num) are included in validation/test set
                if (self.query_per_class == -1):#take all images
                        #print("take all images")
                        self.test_roots += samples[c][train_num:]                    
                else:#in each folder only query_per_class images are included in validation/test set
                      self.test_roots += samples[c][train_num:train_num+self.query_per_class]#takes only  query_per_class query image like relation network


            #print([self.get_class(x) for x in self.train_roots])
            #train_labels and test_labels contain the labels aof sample and query images
            if(len(self.train_roots)>0):#Adding this condition to check for empty training set as in case of testing we may take validation images of training classes from another folder(completely)
                self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
            #if(split == 'test'):
            if(train_num!=self.images_per_class):#Adding this conition to handle the case where test roots are empty
                if(con_test_labels):#added to give continuous labels for train and test.test class lables will start from 1200 in case of omniglot
                    self.test_labels = [labels[self.get_class(x)]+num_train_classes for x in self.test_roots]
                    #print([labels[self.get_class(x)]+num_train_classes for x in self.test_roots])
                    #print("assigning cont labels from ",num_train_classes)
                else:#here test labels will start from zero
                    #print("labels from zero")
                    self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]
        
      
    def get_class(self, sample):
        if(self.class_name_folders==1):
            return  sample.split('/')[-2]
        elif(self.class_name_folders==3):
            return (sample.split('/')[-4]).split('_')[-1]+'_'+sample.split('/')[-2]
        else:
            return sample.split('/')[-3]+ '_' + sample.split('/')[-2]

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transforms=None, target_transform=None):
        self.transforms = transforms # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

#custom dataset for Omniglot        
class OmniglotDataset(FewShotDataset):
    IMAGE_SIZE = 28   # Size of Omniglot images as in state-of-the-art FSL methods. Change for other resolutions
    def __init__(self, *args, **kwargs):
        super(OmniglotDataset, self).__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.image_roots)
    
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L') # To convert color image to gray scale
        image = image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE), resample=Image.LANCZOS) # per Chelsea's implementation; LANCZOS-resampling filter
        #image = np.array(image, dtype=np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[idx]
        """
        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        return image, label
    
    def get_image_path(self):
        return self.image_roots
    
    @staticmethod
    def get_folders(data_folder,num_train=1200):
        
        character_folders = [os.path.join(data_folder, family, character) \
                    for family in os.listdir(data_folder) \
                    if os.path.isdir(os.path.join(data_folder, family))  \
                    for character in os.listdir(os.path.join(data_folder, family))\
                    if os.path.isdir(os.path.join(data_folder, family, character))]
        character_folders.sort()      #Added to avoid the change in the values of meta_train and Test_character_folders during different run  
        random.seed(1)
        random.shuffle(character_folders)

        #num_train = 1200
        
        metatrain_character_folders = character_folders[:num_train]
        metaval_character_folders = character_folders[num_train:]
        
        #creating a dictionary with class mnames as key and numbers as labels for training alphabets
        class_folders=[]
        class_folders.extend(metatrain_character_folders)
        class_folders.sort()
        class_folders[:] = map(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1],class_folders)
        omniglot_train_class_labels_dict=np.array(range(len(class_folders)))
        omniglot_train_class_labels_dict = dict(zip(class_folders, omniglot_train_class_labels_dict))
        
        #creating a dictionary with class names as key and numbers as labels for testing alphabets
        class_folders=[]
        class_folders.extend(metaval_character_folders)
        class_folders.sort()
        class_folders[:] = map(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1],class_folders)
        omniglot_test_class_labels_dict=np.array(range(len(class_folders)))
        omniglot_test_class_labels_dict = dict(zip(class_folders, omniglot_test_class_labels_dict))
        
        return metatrain_character_folders,metaval_character_folders,omniglot_train_class_labels_dict,omniglot_test_class_labels_dict
        

    
    


#For HPL-Tamil dataset    
class OcrHplTamilDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(OcrHplTamilDataset, self).__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.image_roots)
    
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L') # To convert color image to gray scale
        #image = image.resize((32,32), resample=Image.LANCZOS) # per Chelsea's implementation; LANCZOS-resampling filter
        #image = np.array(image, dtype=np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[idx]
        """
        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        return image, label
    
    def get_image_path(self):
        return self.image_roots
    
    #Added is_train to take training class test samples from a different folder
    #Added num_classes to choose different train/test class lists based on the train/test split
    #Added is_load_few_samples (True) to take only few samples for unseeen classes in supervised training case to compare zsl with supervised models trained with few samples of rare classes
    @staticmethod
    def get_folders(data_folder,num_train=-1,is_train=1,num_classes=126,is_load_few_samples=False):
        train_classes=[]
        test_classes=[]
        #print("ia am called")
        if(num_classes == 125):
            print("Train/test split:Corpus 125/30,33")
            trainclasses_fname = 'trainclasses_freq_corpus_old_125.txt'
            testclasses_fname = 'test_classes_rare_corpus_30.txt'
            #testclasses_fname = 'test_classes_rare_corpus_33_with_old.txt'
        elif(num_classes == 127):
            print("Train/test split:Corpus 127/28,31")
            trainclasses_fname = 'trainclasses_freq_corpus_127.txt'
            testclasses_fname = 'test_classes_rare_corpus_28.txt'
           
        elif(num_classes == 126):
            print("Train/test split:User defined 126/29,32")
            trainclasses_fname = 'trainclasses_freq_old_126.txt'
            testclasses_fname = 'testclasses_rare_29.txt'
            
        #for the hyper parameter tuning in with 24 training classes
        elif(num_classes == 24):
            print("Train/test split:User defined 24/131")
            trainclasses_fname = 'trainclasses_subset_24.txt'
            testclasses_fname = 'testlclasses_of_tr24.txt'
        elif(num_classes == 155):#Added to handle supervised training
            print("Supervised training with 155")
            if(not is_load_few_samples):
                trainclasses_fname = 'totalclasses_155.txt'   
            else:#Added this part to handle data loading from two different folders for supervised training of supervised models
                trainclasses_fname = 'trainclasses_freq_old_126.txt'   
                trainclasses_few_sample_fname = 'testclasses_rare_29.txt'         
            testclasses_fname = None
        else:
            print("Warning: Number of classes doesn't match the list of allowed values.\n Taking the list from default file pairs-trainclasses_freq,testclasses_rare")
            trainclasses_fname = 'trainclasses_freq.txt'
            testclasses_fname = 'testclasses_rare.txt'
        #with open(os.path.join(data_folder,'trainclasses_freq.txt')) as f:
        with open(os.path.join(data_folder,trainclasses_fname)) as f:
            for line in f:
                train_classes.append(line.split('\t')[0].strip())
        
        if(is_load_few_samples):
            train_classes_with_few_samples=[]
            with open(os.path.join(data_folder,trainclasses_few_sample_fname)) as f:
                for line in f:
                    train_classes_with_few_samples.append(line.split('\t')[0].strip())
                print("classes with few sampeles",train_classes_with_few_samples)
        #with open(os.path.join(data_folder,'testclasses_rare.txt')) as f:
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            with open(os.path.join(data_folder,testclasses_fname)) as f:
                for line in f:
                    test_classes.append(line.split('\t')[0].strip())
        
        
        
        #Added if condition to fetch the test samples for train classes from the iwfhr test set(iwfhr_Test) during testing in case of GZSL. During training stage they are taken from iwfhr train set(chars_all). Have done this to compare the result of our models with supervised Tamil HCR.
        if(is_train):
            metatrain_character_folders = [os.path.join(data_folder, 'chars_all',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'chars_all'), class_folder))]
            if(is_load_few_samples):
                #print("adding only few training samples for rare chars")
                #without data augmentation
                """
                metatrain_character_folders.extend([os.path.join(data_folder, 'rare_chars_few_samples',class_folder) \
                        for class_folder in train_classes_with_few_samples\
                        if os.path.isdir(os.path.join(os.path.join(data_folder, 'rare_chars_few_samples'), class_folder))])
                """
                #with data augmentation
                metatrain_character_folders.extend([os.path.join(data_folder, 'rare_chars_few_samples_augmented',class_folder) \
                        for class_folder in train_classes_with_few_samples\
                        if os.path.isdir(os.path.join(os.path.join(data_folder, 'rare_chars_few_samples_augmented'), class_folder))])
        else:
            metatrain_character_folders = [os.path.join(data_folder, 'iwfhr_Test',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'iwfhr_Test'), class_folder))]
           
        
       
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            metatest_character_folders = [os.path.join(data_folder, 'iwfhr_Test',class_folder) \
                    for class_folder in test_classes \
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'iwfhr_Test'), class_folder))]
        
        
        
        #creating a dictionary with class mnames as key and numbers as labels for training alphabets
        class_folders=[]
        class_folders.extend(metatrain_character_folders)
        class_folders.sort()
        #print("class_folders after first sort",class_folders)
        class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
        #Adding this condition as in case of loading only few samples from another folder(rare_chars_few_samples changes the order of sorting based on class folders as the training classes come from two different folders. It leads to wrong mapping during testimg where all the samples are taken from single test folder. To avoid this sorting is applied after seperating the class folder numbers from the complete path. Though this will work for all cases to avoid any issues with the earlier scripts adding it only for this particular suoervised training with imbalance and incorrect dataset )
        if(is_load_few_samples):
            class_folders.sort()
            #print("class_folders after second sort",class_folders)
        ocr_hcl_train_class_labels_dict=np.array(range(len(class_folders)))
        ocr_hcl_train_class_labels_dict = dict(zip(class_folders, ocr_hcl_train_class_labels_dict))
        
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            #creating a dictionary with class names as key and numbers as labels for testing alphabets
            class_folders=[]
            class_folders.extend(metatest_character_folders)
            class_folders.sort()
            class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
            ocr_hcl_test_class_labels_dict=np.array(range(len(class_folders)))
            ocr_hcl_test_class_labels_dict = dict(zip(class_folders, ocr_hcl_test_class_labels_dict))

            return metatrain_character_folders,metatest_character_folders,ocr_hcl_train_class_labels_dict,ocr_hcl_test_class_labels_dict
        else:
            return metatrain_character_folders,ocr_hcl_train_class_labels_dict
        
class OcrHplTeluguDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(OcrHplTeluguDataset, self).__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.image_roots)
    
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L') # To convert color image to gray scale
        #image = image.resize((32,32), resample=Image.LANCZOS) # per Chelsea's implementation; LANCZOS-resampling filter
        #image = np.array(image, dtype=np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[idx]
        """
        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        return image, label
    
    def get_image_path(self):
        return self.image_roots
    
       
    @staticmethod
    def get_folders(data_folder,num_train=-1,is_train=1,num_classes=120):
        
        train_classes=[]
        test_classes=[]
        if(num_classes == 120):      
            """
            print("Train/test split: 120/30")
            
            print("User defined split based on features")
            #Split used in TALLIP paper result
            trainclasses_fname = 'trainclasses_120.txt'
            testclasses_fname = 'testclasses_30.txt'
            
            """
            
            #Adding the files for rare character based train/test split of 140 classes
            
            print("Train/test split: 120/20")
            print("Using train/test split based on base classes")
            
            trainclasses_fname = 'trainclasses_freq_140_120.txt'
            testclasses_fname = 'testclasses_rare_140_20.txt'
            
        #Adding the files for rare character based train/test split on new set of 140 classes after removing duplicates          
        elif(num_classes == 110):
            print("Using train/test split based on frequency of occurence")
            print("Train/test split: 110/30")
            
            trainclasses_fname = 'trainclasses_freq_140_110.txt'
            testclasses_fname = 'testclasses_rare_140_30.txt'
            
        elif(num_classes == 150):#Added to handle supervised training
            print("Supervised training with 150")
            trainclasses_fname = 'totalclasses_150.txt'
            testclasses_fname = None
        elif(num_classes == 140):#Added to handle supervised training
            print("Supervised training with 140")
            trainclasses_fname = 'totalclasses_140.txt'
            testclasses_fname = None
        with open(os.path.join(data_folder,trainclasses_fname)) as f:
            for line in f:
                train_classes.append(line.split('\t')[0].strip())
                
        if(testclasses_fname is not None) :  #Added if condition to handle supervised training    
            #with open(os.path.join(data_folder,'testclasses_rare.txt')) as f:
            with open(os.path.join(data_folder,testclasses_fname)) as f:
                for line in f:
                    test_classes.append(line.split('\t')[0].strip())                
        #Added if condition to fetch the test samples for train classes from the iwfhr test set(iwfhr_Test) during testing in case of GZSL. During training stage they are taken from iwfhr train set(chars_all). Have done this to compare the result of our models with supervised Tamil HCR.
        if(is_train):
            metatrain_character_folders = [os.path.join(data_folder, 'chars_all_train',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'chars_all_train'), class_folder))]
        else:
            metatrain_character_folders = [os.path.join(data_folder, 'chars_all_test',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'chars_all_test'), class_folder))]
        if(testclasses_fname is not None):#Added if condition to handle supervised training  
            metatest_character_folders = [os.path.join(data_folder, 'chars_all_test',class_folder) \
                        for class_folder in test_classes \
                        if os.path.isdir(os.path.join(os.path.join(data_folder, 'chars_all_test'), class_folder))]
        
        #creating a dictionary with class mnames as key and numbers as labels for training alphabets
        class_folders=[]
        class_folders.extend(metatrain_character_folders)
        class_folders.sort()
        class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
        ocr_hcl_train_class_labels_dict=np.array(range(len(class_folders)))
        ocr_hcl_train_class_labels_dict = dict(zip(class_folders, ocr_hcl_train_class_labels_dict))
        
        if(testclasses_fname is not None):#Added if condition to handle supervised training  
            #creating a dictionary with class names as key and numbers as labels for testing alphabets
            class_folders=[]
            class_folders.extend(metatest_character_folders)
            class_folders.sort()
            class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
            ocr_hcl_test_class_labels_dict=np.array(range(len(class_folders)))
            ocr_hcl_test_class_labels_dict = dict(zip(class_folders, ocr_hcl_test_class_labels_dict))
        
            return metatrain_character_folders,metatest_character_folders,ocr_hcl_train_class_labels_dict,ocr_hcl_test_class_labels_dict
        else:
            return metatrain_character_folders,ocr_hcl_train_class_labels_dict
    


#Modified IndicSixPrintedDataset dataset class(of FSL) to use different split for zsl training.
class IndicSixPrintedDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(IndicSixPrintedDataset, self).__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.image_roots)
    
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root).convert('RGB')
        #image = Image.open(image_root).convert('L')
        if self.transforms is not None:
            #print("applying transforms")
            image = self.transforms(image)
        label = self.labels[idx]
        
        """
        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        return image, label
    
    def get_image_path(self):
        return self.image_roots
    
    #Adding is_zsl attribute to use the indic script for zsl training with different trainig test split
    @staticmethod
    def get_folders(data_folder,num_train=800,is_zsl=0,num_classes = 126):
        if(is_zsl):
            train_classes=[]
            test_classes=[]
            if(num_classes == 125):
                print("Train/test split:Corpus 125/30")
                trainclasses_fname = 'trainclasses_freq_corpus_old_125.txt'
                testclasses_fname = 'test_classes_rare_corpus_30.txt'
            elif(num_classes == 127):
                print("Train/test split:Corpus 127/28")
                trainclasses_fname = 'trainclasses_freq_corpus_127.txt'
                testclasses_fname = 'test_classes_rare_corpus_28.txt'
            elif(num_classes == 126):
                print("Train/test split:User defined 126/29")
                trainclasses_fname = 'trainclasses_freq_old_126.txt'
                testclasses_fname = 'testclasses_rare_29.txt'
            elif(num_classes == 155):#Added to handle supervised training
                print("Supervised training with 155")
                trainclasses_fname = 'totalclasses_155.txt'
                testclasses_fname = None
            else:
                print("Warning: Number of classes doesn't match the list of allowed values.\n Taking the list from default file pairs-trainclasses_freq,testclasses_rare")
                trainclasses_fname = 'trainclasses_freq.txt'
                testclasses_fname = 'testclasses_rare.txt'
            with open(os.path.join(data_folder,trainclasses_fname)) as f:
                for line in f:
                    train_classes.append(line.split('\t')[0].strip())

            if(testclasses_fname is not None):#Added if condition to handle supervised training 
                with open(os.path.join(data_folder,testclasses_fname)) as f:
                    for line in f:
                        test_classes.append(line.split('\t')[0].strip())
            
            metatrain_character_folders = [os.path.join(data_folder, 'Tamil',class_folder) \
                            for class_folder in train_classes\
                            if os.path.isdir(os.path.join(os.path.join(data_folder, 'Tamil'), class_folder))]

            if(testclasses_fname is not None):#Added if condition to handle supervised training 
                metatest_character_folders = [os.path.join(data_folder, 'Tamil',class_folder) \
                            for class_folder in test_classes \
                            if os.path.isdir(os.path.join(os.path.join(data_folder, 'Tamil'), class_folder))]
             
        else:
            character_folders = [os.path.join(data_folder, family, character) \
                        for family in os.listdir(data_folder) \
                        if os.path.isdir(os.path.join(data_folder, family)) \
                        for character in os.listdir(os.path.join(data_folder, family))\
                        if os.path.isdir(os.path.join(data_folder, family, character))]
            character_folders.sort()      #Added to avoid the change in the values of meta_train and Test_character_folders during different run  
            random.seed(1)
            random.shuffle(character_folders)

            #num_train = 1200

            metatrain_character_folders = character_folders[:num_train]
            metatest_character_folders = character_folders[num_train:]
        
        #creating a dictionary with class mnames as key and numbers as labels for training alphabets
        class_folders=[]
        class_folders.extend(metatrain_character_folders)
        class_folders.sort()
        if(is_zsl):
            class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
        else:
            class_folders[:] = map(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1],class_folders)
        indic_train_class_labels_dict=np.array(range(len(class_folders)))
        indic_train_class_labels_dict = dict(zip(class_folders, indic_train_class_labels_dict))
        
        if(not is_zsl or testclasses_fname is not None):#Added if condition to handle supervised training 
            #creating a dictionary with class names as key and numbers as labels for testing alphabets
            class_folders=[]
            class_folders.extend(metatest_character_folders)
            class_folders.sort()
            if(is_zsl):
                class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
            else:
                class_folders[:] = map(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1],class_folders)
            indic_test_class_labels_dict=np.array(range(len(class_folders)))
            indic_test_class_labels_dict = dict(zip(class_folders, indic_test_class_labels_dict))
        
            return metatrain_character_folders,metatest_character_folders,indic_train_class_labels_dict,indic_test_class_labels_dict
        else:
            return metatrain_character_folders,indic_train_class_labels_dict




class uTHCDDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(uTHCDDataset, self).__init__(*args, **kwargs)
    
    def __len__(self):
        return len(self.image_roots)
    
    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L') # To convert color image to gray scale
        #image = image.resize((32,32), resample=Image.LANCZOS) # per Chelsea's implementation; LANCZOS-resampling filter
        #image = np.array(image, dtype=np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.labels[idx]
        """
        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        return image, label
    
    def get_image_path(self):
        return self.image_roots
    
    @staticmethod
    def get_folders(data_folder,num_train=-1,num_classes =126):
        train_classes=[]
        test_classes=[]
        if(num_classes == 125):
            print("Train/test split:Corpus 125/30")
            trainclasses_fname = 'trainclasses_freq_corpus_125.txt'
            testclasses_fname = 'testclasses_rare_corpus_30.txt'
        elif(num_classes == 127):
            print("Train/test split:Corpus 127/28")
            trainclasses_fname = 'trainclasses_freq_corpus_127.txt'
            testclasses_fname = 'testclasses_rare_corpus_28.txt'
        elif(num_classes == 126):
            print("Train/test split:User defined 126/29")
            trainclasses_fname = 'trainclasses_freq_old_126.txt'
            testclasses_fname = 'testclasses_rare_29.txt'
        elif(num_classes == 155):#Added to handle supervised training
            print("Supervised training with 155")
            trainclasses_fname = 'totalclasses_155.txt'
            testclasses_fname = None
        else:
            print("Warning: Number of classes doesn't match the list of allowed values.\n Taking the list from default file pairs-trainclasses_freq,testclasses_rare")
            trainclasses_fname = 'trainclasses_freq.txt'
            testclasses_fname = 'testclasses_rare.txt'
        #with open(os.path.join(data_folder,'trainclasses_freq.txt')) as f:
        with open(os.path.join(data_folder,trainclasses_fname)) as f:
            for line in f:
                train_classes.append(line.split('\t')[0].strip())
                
        #with open(os.path.join(data_folder,'testclasses_rare.txt')) as f:
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            with open(os.path.join(data_folder,testclasses_fname)) as f:
                for line in f:
                    test_classes.append(line.split('\t')[0].strip())        
        
        
        #Modifying to adjust the number of training and testclasses using the list of classes stored in text files
        metatrain_character_folders = [os.path.join(data_folder, 'chars_all',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'chars_all'), class_folder))]
        
        metaval_character_folders = [os.path.join(data_folder, 'val_all',class_folder) \
                    for class_folder in train_classes\
                    if os.path.isdir(os.path.join(os.path.join(data_folder, 'val_all'), class_folder))]
        
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            #For test folder we take the test samples(180) of test classes. 
            metatest_character_folders = [os.path.join(data_folder, 'val_all',class_folder) \
                        for class_folder in test_classes\
                        if os.path.isdir(os.path.join(os.path.join(data_folder, 'val_all'), class_folder))]

        
        
        
        #creating a dictionary with class mnames as key and numbers as labels for training classes
        class_folders=[]
        class_folders.extend(metatrain_character_folders)
        class_folders.sort()
        class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
        uTHCD_train_class_labels_dict=np.array(range(len(class_folders)))
        uTHCD_train_class_labels_dict = dict(zip(class_folders, uTHCD_train_class_labels_dict))
        
        #creating a dictionary with class names as key and numbers as labels for validation classes
        class_folders=[]
        class_folders.extend(metaval_character_folders)
        class_folders.sort()
        class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
        uTHCD_val_class_labels_dict=np.array(range(len(class_folders)))
        uTHCD_val_class_labels_dict = dict(zip(class_folders, uTHCD_val_class_labels_dict))
        
        if(testclasses_fname is not None):#Added if condition to handle supervised training 
            #creating a dictionary with class names as key and numbers as labels for testing alphabets
            class_folders=[]
            class_folders.extend(metatest_character_folders)
            class_folders.sort()
            class_folders[:] = map(lambda x: x.split('/')[-1],class_folders)
            uTHCD_test_class_labels_dict=np.array(range(len(class_folders)))
            uTHCD_test_class_labels_dict = dict(zip(class_folders, uTHCD_test_class_labels_dict))

            return metatrain_character_folders,metaval_character_folders,metatest_character_folders,uTHCD_train_class_labels_dict,uTHCD_val_class_labels_dict,uTHCD_test_class_labels_dict
        else:
            return metatrain_character_folders,metaval_character_folders,uTHCD_train_class_labels_dict,uTHCD_val_class_labels_dict



        
#Added for Malayalam character dataset stored in .csv file

class MalayalamCSVDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None, is_invert = False):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.is_invert = is_invert
        # keep only rows whose label is in mapping (robustness)
        self.df = self.df[self.df.iloc[:, 0].isin(self.class_to_idx)].reset_index(drop=True)
        
        # optional sanity checks
        if len(self.df) == 0:
            raise ValueError("After filtering, dataset is empty. Check your class lists vs CSV labels.")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # first column = label
        #label = self.class_to_idx[str(row.iloc[0])]
        # first column = label
        label = self.class_to_idx[int(row.iloc[0])] #integer label

        # remaining columns = pixels
        pixels = row.iloc[1:].to_numpy(dtype=np.uint8)
        
        

        # reshape into 32x32 column-major
        img_array = pixels.reshape((32, 32), order="F")
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()  # (1,32,32)

        #Added to invert the images to have white background to test them with 
        if(self.is_invert):
            img_tensor = 1- img_tensor
        # convert to PIL Image
        #img = Image.fromarray(img_array, mode="L")   # grayscale
        # If model expects RGB:
        # img = img.convert("RGB")

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
   

    @staticmethod
    def get_data(
        datafilepath=None,
        train_csv="train.csv",
        val_csv="val.csv",
        test_csv="test.csv",
        mode="zsl",  # "zsl" -> test only unseen; "gzsl" -> test uses all classes present
        is_train = 1
    ):
        
        if datafilepath is None:
            raise ValueError("datafilepath must be given: the folder containing trainclasses.txt, "
                         "testclasses.txt and the train/val/test CSV files")
        seen_txt=os.path.join(datafilepath,"trainclasses.txt")
        unseen_txt=os.path.join(datafilepath,"testclasses.txt")
        
        
        # read class lists
        seen_classes = read_classes(seen_txt)
        unseen_classes = read_classes(unseen_txt)
        
        train_csv = os.path.join(datafilepath,train_csv)
        val_csv = os.path.join(datafilepath,val_csv)
        test_csv = os.path.join(datafilepath,test_csv)

        # sanity: warn if overlap (pure ZSL expects none)
        overlap = set(seen_classes).intersection(unseen_classes)
        if overlap :
            print(f"[WARN] Overlap between seen and unseen classes: {sorted(overlap)}")

        # load CSVs and filter
        train_df = load_and_filter_csv(train_csv, allowed=set(seen_classes))
        val_df   = load_and_filter_csv(val_csv,   allowed=set(seen_classes))

        if mode == "zsl":
            # test restricted to unseen classes
            test_df = load_and_filter_csv(test_csv, allowed=set(unseen_classes))
            test_classes_for_mapping = unseen_classes
            train_class_label_dict = build_class_to_idx(seen_classes)
            test_class_label_dict = build_class_to_idx(test_classes_for_mapping)

        elif mode == "gzsl":
            # test uses whatever is in test.csv (no filtering)
            test_df = load_and_filter_csv(test_csv, allowed=None)
            combined_class_label_dict, train_class_label_dict, test_class_label_dict = \
            build_gzsl_class_to_idx(seen_classes, unseen_classes)
            
        elif mode == "sup":
            test_df = load_and_filter_csv(test_csv, allowed=set(seen_classes))
            test_classes_for_mapping = seen_classes
            train_class_label_dict = build_class_to_idx(seen_classes)
            test_class_label_dict = build_class_to_idx(test_classes_for_mapping)
        else:
            raise ValueError("mode must be 'zsl','gzsl' or 'sup'")

        

        if(is_train):
            return train_class_label_dict,test_class_label_dict,train_df,val_df
        else:
            if(mode=="gzsl"):
                return train_class_label_dict,test_class_label_dict,combined_class_label_dict, test_df
            else:
                return train_class_label_dict,test_class_label_dict,test_df



#Dataset with mapped labels of other dataset
#Used for cross dataset evaluation
class LabelMappedDataset(Dataset):
    def __init__(self, base_dataset, label_id_map):
        self.base_dataset = base_dataset
        self.label_id_map = label_id_map

    def __getitem__(self, idx):
        img, old_label = self.base_dataset[idx]
        #Modified to handle the error of KanndaHW cross dataset execution to convert the tensor to integer item
        #new_label = self.label_id_map[old_label]
        new_label = self.label_id_map[old_label.item()]
        return img, new_label

    def __len__(self):
        return len(self.base_dataset)

    

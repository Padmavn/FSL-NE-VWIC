"""
DE_AutoEncoder: a convolutional variational auto-encoder that separates the
representation of an image into a *content* code and a *style* code, with a
cosine-similarity classifier attached to the content code.

The overall design (an auto-encoder whose latent code is split into a content part and a style
part, used for few-shot classification) is based on the master's thesis "Few-shot Classification
by Learning Disentangled Representations". No source code was available for that work, so this
is an independent implementation, not a copy. It differs from the thesis in that it uses a
Beta-VAE and a cosine-similarity classifier trained with a different loss, instead of a standard
VAE with an episodic classifier.

Architecture
  * Encoder    : 3 or 4 conv layers (3x3, padding 1). Each is applied as conv -> BatchNorm -> max-pool
                 (2x2, ceil_mode=True) -> ReLU; the max-pool is skipped only for the last layer of a
                 4-layer encoder when `is_max_pool_after_last_conv=0`. The flattened features pass
                 through dropout and then through four linear layers that give the mean and log-variance
                 of the content distribution and of the style distribution (reparameterization trick
                 of the Beta-VAE).
  * Decoder    : a linear layer followed by transposed convolutions. It takes the concatenated
                 [content, style] samples and returns the reconstructed image as raw values
                 (no sigmoid), so the reconstruction loss should use logits, e.g. BCE-with-logits.
  * Classifier : a single bias-free linear layer on the L2-normalised content code, i.e. a
                 cosine-similarity classifier. The normalised code can be multiplied by a trainable
                 scalar (`scale`, initial value 10), following the imprinted-weights approach
                 (Qi et al., "Low-Shot Learning with Imprinted Weights", CVPR 2018). The in-place
                 weight normalisation of the classifier (`pure_norm=False`) is adapted from the
                 `weight_norm` method of https://github.com/YU1ut/imprinted-weights, an unofficial
                 PyTorch implementation of that paper.

Args
  code_size                  : [content_dim, style_dim]. code_size[0] is the content dimension (also the
                               classifier input). code_size[1] is the style dimension and is read only
                               when `is_same_style_dim=0`.
  input_size                 : height/width of the square input image (28 for Omniglot).
  no_of_in_channels,
  no_of_out_channels         : per-layer conv channels; their length (3 or 4) sets the depth.
  num_classes                : number of training classes (outputs of the classifier).
  is_max_pool_after_last_conv: affects only the last (4th) conv layer of a 4-layer encoder:
                               1 = it is max-pooled too; 0 = it is not (the setting used for Omniglot).
                               The first three layers always pool, so the flag has no effect on a 3-layer encoder.
  drop_prob                  : dropout probability on the flattened encoder features.
  scale                      : multiply the normalised content code by the trainable scalar `self.s`.
  is_same_style_dim          : 1 = style code has the same size as the content code;
                               0 = style size is code_size[2].
  is_train                   : 1 = forward() also decodes and returns the reconstruction;
                               0 = decoder is skipped (testing). This is a constructor flag and is
                               independent of nn.Module.train() / eval().
  pure_norm                  : False (default) = classifier weights are normalised in place before
                               each forward pass, as in the original implementation used for the
                               published results; True = normalisation is part of the computation graph.

forward() returns
  is_train=1 : (reconstruction, class_scores, mu_content, logvar_content, mu_style, logvar_style);
               the class scores are computed from the sampled content code.
  is_train=0 : (class_scores, mu_content, logvar_content, mu_style, logvar_style);
               the class scores are computed from the mean of the content code.

Supported sizes: the decoder output must equal `input_size` or be at least `input_size + 3`
(it is then cropped); otherwise a ValueError is raised. With 4 layers and
is_max_pool_after_last_conv=0 this holds for 28, 50 and 105 but not for 32 or 64
(use is_max_pool_after_last_conv=1 or a 3-layer encoder for those).

The constructor raises a ValueError if the number of conv layers is not 3 or 4, if
no_of_in_channels and no_of_out_channels differ in length or do not chain
(no_of_in_channels[i+1] == no_of_out_channels[i]), or if code_size has no style entry when
is_same_style_dim=0.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DE_AutoEncoder(nn.Module):    
    def __init__(self, code_size=[1024],input_size=28,no_of_in_channels=[1,128,256,512],no_of_out_channels=[128,256,512,1024],num_classes=1200,is_max_pool_after_last_conv=0,drop_prob=0.25,scale=True,is_same_style_dim=1,is_train=1,pure_norm= False):
        n_layers = len(no_of_in_channels)
        if n_layers not in (3, 4):
            raise ValueError(f"The model supports 3 or 4 conv layers, got {n_layers}.")
        if len(no_of_out_channels) != n_layers:
            raise ValueError("no_of_in_channels and no_of_out_channels must have the same length "
                             f"({n_layers} vs {len(no_of_out_channels)}).")
        for i in range(n_layers - 1):
            if no_of_in_channels[i + 1] != no_of_out_channels[i]:
                raise ValueError(f"no_of_in_channels[{i + 1}] ({no_of_in_channels[i + 1]}) must equal "
                                 f"no_of_out_channels[{i}] ({no_of_out_channels[i]}) because layer {i + 2} "
                                 f"takes the output of layer {i + 1}.")
        if not is_same_style_dim and len(code_size) < 2:
            raise ValueError("code_size must be [content_dim, style_dim] when is_same_style_dim=0.")
        super().__init__()
        self.code_size = code_size #output dimensions of the linear layers
        self.input_size= input_size #input size (For 32*32 image give 32 )
        self.no_of_in_channels=no_of_in_channels #List that has the no of filters used in each conv. layer)
        self.no_of_out_channels=no_of_out_channels #List that has the no of filters used in each conv. layer)
        self.no_of_layers = len(no_of_in_channels)
        self.num_classes = num_classes
        self.is_max_pool_after_last_conv = is_max_pool_after_last_conv
        self.scale = scale
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.is_train = is_train
        self.pure_norm = pure_norm
       
        self.enc_cnn_1 = nn.Conv2d(self.no_of_in_channels[0], self.no_of_out_channels[0], kernel_size=3,padding=1)
        self.enc_cnn_2 = nn.Conv2d(self.no_of_in_channels[1],self.no_of_out_channels[1], kernel_size=3,padding=1)
        self.enc_cnn_3 = nn.Conv2d(self.no_of_in_channels[2],self.no_of_out_channels[2], kernel_size=3,padding=1)
        if(self.no_of_layers==4):
            self.enc_cnn_4 = nn.Conv2d(self.no_of_in_channels[3], self.no_of_out_channels[3], kernel_size=3,padding=1)
        
       
        self.bn1 = nn.BatchNorm2d(self.no_of_out_channels[0])
        self.bn2 = nn.BatchNorm2d(self.no_of_out_channels[1])
        self.bn3 = nn.BatchNorm2d(self.no_of_out_channels[2])
        if(self.no_of_layers==4):
            self.bn4 = nn.BatchNorm2d(self.no_of_out_channels[3])
        
        # To handle different input size
        # output size (for different input size) after the encoder's ceil_mode max-pools: 
        # ceil(n / 2^k),where k = no_of_layers (3-layer model, or 4 layers with last-layer maxpool on) or no_of_layers - 1 (4 layers, maxpool off).
        if(self.no_of_layers<4 or self.is_max_pool_after_last_conv):
            self.input_to_linear = math.ceil(self.input_size/pow(2,self.no_of_layers))
            
        else:#omniglot case
            self.input_to_linear = math.ceil(self.input_size/pow(2,self.no_of_layers-1))
        #print("o/p channels:",self.input_to_linear)
        
        self.input_to_linear_code_size = self.no_of_out_channels[self.no_of_layers-1]*self.input_to_linear*self.input_to_linear
        
        #For obtaining the mu and var of content
        self.enc_linear_1 = nn.Linear(self.input_to_linear_code_size, self.code_size[0])
        self.enc_linear_2 = nn.Linear(self.input_to_linear_code_size, self.code_size[0])

        #For obtaining the mu and var of style
        if(is_same_style_dim):
            self.enc_linear_3 = nn.Linear(self.input_to_linear_code_size, self.code_size[0])
            self.enc_linear_4 = nn.Linear(self.input_to_linear_code_size, self.code_size[0])
        else:
            self.enc_linear_3 = nn.Linear(self.input_to_linear_code_size, self.code_size[1])
            self.enc_linear_4 = nn.Linear(self.input_to_linear_code_size, self.code_size[1])
        
        if(is_same_style_dim):
            self.dec_linear_1 = nn.Linear(self.code_size[0]*2, self.input_to_linear_code_size) 
        else:
            self.dec_linear_1 = nn.Linear(self.code_size[0]+self.code_size[1], self.input_to_linear_code_size)
        
        self.dec_cnn_1 = nn.ConvTranspose2d( self.no_of_out_channels[self.no_of_layers-1],  self.no_of_in_channels[self.no_of_layers-1], kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_cnn_2 = nn.ConvTranspose2d(self.no_of_out_channels[self.no_of_layers-2], self.no_of_in_channels[self.no_of_layers-2], kernel_size=3,stride=2,padding=1,output_padding=1)
        self.dec_cnn_3 = nn.ConvTranspose2d(self.no_of_out_channels[self.no_of_layers-3], self.no_of_in_channels[self.no_of_layers-3], kernel_size=3,stride=2,padding=1,output_padding=1)
        if(self.no_of_layers==4):
            if(is_max_pool_after_last_conv):
                self.dec_cnn_4 = nn.ConvTranspose2d(self.no_of_out_channels[self.no_of_layers-4], self.no_of_in_channels[self.no_of_layers-4], kernel_size=3,stride=2,padding=1,output_padding=1)
            else:
                self.dec_cnn_4 = nn.ConvTranspose2d(self.no_of_out_channels[self.no_of_layers-4], self.no_of_in_channels[self.no_of_layers-4], kernel_size=3,stride=1,padding=0,output_padding=0)
                
        #Single classifier layer with bias as False 
        self.class_linear_2 = nn.Linear(self.code_size[0],self.num_classes,bias=False)
        
        self.dropout_layer =nn.Dropout(p=drop_prob)
        
    #Reparameterization trick of Beta-VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    #Forward pass through the Linear layers of Beta-VAE to generate the data distribution parameters
    def bottleneck(self, h):
        mu_con, logvar_con = self.enc_linear_1(h), self.enc_linear_2(h)
        z_con = self.reparameterize(mu_con, logvar_con)
        mu_sty, logvar_sty = self.enc_linear_3(h), self.enc_linear_4(h)
        z_sty = self.reparameterize(mu_sty, logvar_sty)
        return mu_con, logvar_con,z_con, mu_sty, logvar_sty, z_sty 
        
    
    #Forward pass of the encoder of Beta-VAE
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = self.bn1(code) #Batch normalization
        
        code = F.relu(F.max_pool2d(code, 2,ceil_mode=True))
        #code = F.relu(F.max_pool2d(code, 2))
        #print("After cnn 1:",code.size())
        
        code = self.enc_cnn_2(code)
        code = self.bn2(code) #Batch normalization
        code = F.relu(F.max_pool2d(code, 2,ceil_mode=True))
        #code = F.relu(F.max_pool2d(code, 2))
        #print("After cnn 2:",code.size())
        
        code = self.enc_cnn_3(code)
        code = self.bn3(code) #Batch normalization
        code = F.relu(F.max_pool2d(code, 2,ceil_mode=True))
        #code = F.relu(F.max_pool2d(code, 2))
        #print("After cnn 3:",code.size())
        
        if(self.no_of_layers==4):
            code = self.enc_cnn_4(code)
            code = self.bn4(code) #Batch normalization
            if(self.is_max_pool_after_last_conv):
                
                code = F.relu(F.max_pool2d(code, 2,ceil_mode=True))#for omniglot
                #code = F.relu(F.max_pool2d(code, 2))
            else:
                code = F.relu(code)
                #code =code #Trying without relu and maxpool with english dataset
            #print("After cnn 4:",code.size())
        
        
        code = code.view(images.size(0), -1) #Flattening(1024 * 4 * 4)
        #print("After flattening:",code.size())
        
        #Apply dropout layer
        code=self.dropout_layer(code)
        
        mu_content,var_content,z_content,mu_style,var_style,z_style = self.bottleneck(code)
        return mu_content,var_content,z_content,mu_style,var_style,z_style
    
    #Forward pass of the decoder of Beta-VAE
    def decode(self, code):
        
        out = self.dec_linear_1(code)
        
        first_dim_size=out.size()[0]
        out = out.view(first_dim_size,self.no_of_out_channels[self.no_of_layers-1], self.input_to_linear,self.input_to_linear)
        #print("from fc:",out.size())
        
        #adding the missed RELU
        out = F.relu(out)
        
        out = self.dec_cnn_1(out)
        out = F.relu(out)
        #print("from dec_cnn1:",out.size())
        
        out = self.dec_cnn_2(out)
        out = F.relu(out)
        #print("from dec_cnn2:",out.size())
  
        out = self.dec_cnn_3(out)        
        #print("from dec_cnn3:",out.size())
        
        if(self.no_of_layers==4):
            out = F.relu(out) #relu of prev layer
            out = self.dec_cnn_4(out)
           
        
        
        #print("decodert's output before slicing:",out.size())
        
        #To check whether slicing is required
        if(self.input_size!=out.size()[2]):            
            out = out[:,:,3:3+self.input_size,3:3+self.input_size] #slicing operation to handle different input size
        #Give error message when the 3 pixel slicing doesn't work with the given input size
        if out.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"Decoder output is {tuple(out.shape[-2:])} but expected "
                f"({self.input_size}, {self.input_size}). The fixed 3-pixel crop only "
                "supports the input sizes that generate the decoder output same as input size or at least input_size+3."
            )
        
        return out
    
    
    
    #Classifier with a single fc layer that use cosine-similarity based softmax score
    def classifier(self,x):
        x = F.normalize(x, p=2, dim=1)#As we have only one layer in classifier
        if self.scale:
            x = self.s * x

        if self.pure_norm:  # new option, default False
            w = F.normalize(self.class_linear_2.weight, p=2, dim=1)
            x = F.linear(x, w)
        else:
            # Original implementation used for the published results:
            # Updates the normalized weights before each forward pass.
            w = F.normalize(self.class_linear_2.weight.data, p=2, dim=1)
            self.class_linear_2.weight.data = w
            x = self.class_linear_2(x)
        return x
    
    
        
    def reconstruct(self, images):
        mu_content,logvar_content,z_content,mu_style,logvar_style,z_style= self.encode(images)
        code = torch.cat((z_content,z_style),1)
        #print("input to decode(dim1):",code.size())
        out = self.decode(code)
        return out, mu_content,logvar_content,z_content,mu_style,logvar_style
        
    def forward(self, images):
        #print(images.size())
        if(not self.is_train):
            mu_content,logvar_content,z_content,mu_style,logvar_style,z_style= self.encode(images)
            out_class = self.classifier(mu_content)
            return out_class ,mu_content,logvar_content,mu_style,logvar_style
        else:
            out, mu_content,logvar_content,z_content,mu_style,logvar_style = self.reconstruct(images)
            out_class = self.classifier(z_content)
            return out, out_class ,mu_content,logvar_content,mu_style,logvar_style
    
    #To extract the normalized mean vectors of the content distribution generated by the beta-vae
    #def extract_fc(self,images,is_gen_sample=0):
    def extract_fc(self,images):
        mu_content,logvar_content,z_content,mu_style,logvar_style,z_style= self.encode(images)
        # Classifier with single fc 
        out_fc = F.normalize(mu_content, p=2, dim=1) #use when we have only one layer in classifier
        return out_fc
        




 

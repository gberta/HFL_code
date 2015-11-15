# -*- coding: utf-8 -*-
'''
Created on Jun 14, 2014

@author: Gedas
'''
import random
import os
import time
import numpy as np
import gnumpy as gp
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io
from scipy import linalg
import scipy.io
import numpy.matlib
#from scipy.linalg import get_blas_funcs
from scipy.linalg import blas as FB
#import scipy.linalg as sp
import cudamat as cm

#%matplotlib inline

import sys

#print numpy.show_config() 
#sys.exit(1)

def get_probs(a3):
   probs=a3

   LIM=0.8
   (list_rows,list_cols)=np.where(a3 > LIM)
   temp=np.zeros((len(list_rows),1))
   max_val=-999999
   min_val=999999
   for i in range(len(list_rows)):
     cur_val=a3[list_rows[i],0]
     temp[i,0]=cur_val
     if cur_val>max_val:
        max_val=cur_val
     if cur_val<min_val:
        min_val=cur_val 
   r=max_val-min_val
   
   for i in range(len(list_rows)):
     temp[i,0]-=LIM
     temp[i,0]/=r
     temp[i,0]*=(1-LIM)
     temp[i,0]+=LIM
     
     probs[list_rows[i],0]=temp[i,0]

   LIM=0.35
   (list_rows,list_cols)=np.where(a3 < LIM)
   temp=np.zeros((len(list_rows),1))
   max_val=-999999
   min_val=999999
   for i in range(len(list_rows)):
     cur_val=a3[list_rows[i],0]
     temp[i,0]=cur_val
     if cur_val>max_val:
        max_val=cur_val
     if cur_val<min_val:
        min_val=cur_val 
   r=max_val-min_val
   
   for i in range(len(list_rows)):
     temp[i,0]-=min_val
     temp[i,0]/=r
     temp[i,0]*=LIM
     #temp[i,0]+LIM
     
     probs[list_rows[i],0]=temp[i,0]

   return probs

def get_features(filter_blob,orig_cords,orig_dim):
 
    #print 'Get features...\n'
    time1 = time.time()
    #time3 = time.time()

    dist=np.zeros((orig_cords.shape[0],4))
    cords=np.zeros((orig_cords.shape[0],4))
    indices=np.zeros((orig_cords.shape[0],4))

    num_rows=orig_cords.shape[0]

    orig_height=orig_dim[0]
    orig_width=orig_dim[1]

    ch=filter_blob.shape[0]
    cur_height=filter_blob.shape[1]
    cur_width=filter_blob.shape[2]

    orig_row=orig_cords[:,0]
    orig_col=orig_cords[:,1]

    orig_row_norm=(orig_row-1)/float(orig_height)
    orig_col_norm=(orig_col-1)/float(orig_width)

    cur_row=orig_row_norm*cur_height
    cur_col=orig_col_norm*cur_width


    
    
    cords[:,0]=np.clip(np.floor(cur_row),0,cur_height-1)
    cords[:,1]=np.clip(np.ceil(cur_row),0,cur_height-1)
    cords[:,2]=np.clip(np.floor(cur_col),0,cur_width-1)
    cords[:,3]=np.clip(np.ceil(cur_col),0,cur_width-1)

    filters=np.reshape(filter_blob,(filter_blob.shape[0],filter_blob.shape[1]*filter_blob.shape[2]))
    #filters=gp.garray(filters)

    

    dist[:,0]=abs(cur_row-cords[:,0])+abs(cur_col-cords[:,2])
    dist[:,1]=abs(cur_row-cords[:,0])+abs(cur_col-cords[:,3])
    dist[:,2]=abs(cur_row-cords[:,1])+abs(cur_col-cords[:,2])
    dist[:,3]=abs(cur_row-cords[:,1])+abs(cur_col-cords[:,3])

    #print dist.shape
    min_ind=np.argmin(dist,1)
    #min_ind=np.reshape(min_ind,(1,num_rows))
  
    ord=np.arange(num_rows)
    ord=ord*4+min_ind

    indices[:,0]=cords[:,0]*cur_width+cords[:,2]
    indices[:,1]=cords[:,0]*cur_width+cords[:,3]
    indices[:,2]=cords[:,1]*cur_width+cords[:,2]
    indices[:,3]=cords[:,1]*cur_width+cords[:,3]


    indices=np.reshape(indices,(1,num_rows*4))


    temp=indices[:,ord]
    temp=np.reshape(temp,(num_rows,1))

    indices=temp.astype(int)


    unq_indices, I=np.unique(indices,return_inverse=True)
    U=unq_indices.astype(int)


    

       
    vals=filters[:,U]
    #vals=filters[U,:]

       
    #if ch>256:
    vals=gp.garray(vals)
    #vals=np.transpose(vals)
   

    vals=vals[:,I]
   
    vals=vals.T

    return vals



#sys.exit(1)

input_img_file=sys.argv[1] #input image file
output_file=sys.argv[2]


cm.cuda_set_device(0)
cm.cublas_init()


if '/' in input_img_file:
   str_arr=input_img_file.split('/')
   l=len(str_arr)
   arr=str_arr[l-1].split('.')
   name=arr[0]
else:
   arr=input_img_file.split('.')
   name=arr[0]
  
print 'Processing image ' + str(name) + '...'




### SPECIFICATION OF PATHS #######
caffe_root = '/PATH/TO/CAFFE/ROOT/DIR/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '/usr/local/cuda/lib')
import caffe

#deploy_path=caffe_root + 'examples/VGG/VGG_big_deploy.prototxt'
deploy_path=caffe_root + 'examples/VGG/HFL_deploy_fast.prototxt'
VGG_model_path=caffe_root + 'examples/VGG/VGG_model'

############


if os.path.exists(VGG_model_path):
    print 'Loading network...'
    net = caffe.Classifier(deploy_path, VGG_model_path)
else:
    print 'Network file doesnt exist!'
    sys.exit(1)

print 'Loaded network!'

net.set_phase_test()
net.set_mode_gpu()


net.set_mean('data', caffe_root + 'examples/VGG/VGG_big_mean.npy') 

print 'Loaded mean!'

net.set_channel_swap('data', (2,1,0))
net.set_input_scale('data', 255) 


hfl_data=scipy.io.loadmat('hfl_model/HFL.mat')

mu=hfl_data['mu'] 
sigma=hfl_data['sigma'] 

sigma=np.c_[1.0,sigma] #adding bias
mu=np.c_[-1.0,mu] #adding bias



                       
if '.jpg' in input_img_file or '.png' in input_img_file:

    cur_im=caffe.io_caffe.load_image(input_img_file)
    num_rows=cur_im.shape[0]
    num_cols=cur_im.shape[1]
    orig_dim=(num_rows,num_cols)

    input_data=[]
    input_data.append(cur_im)


    print 'Computing candidate points...'
    #generating candidates
    
    init_file=name+'_init_fast.jpg'
    se_model='../../../se_detector/SE_model.yml'
    se_command='../../../se_detector/SE_detector '+str(se_model)+' '+str(input_img_file)+' '+str(init_file)
    os.system(se_command)

    if not os.path.exists(init_file):
     print 'ERROR: No candidate file '+str(init_file)
     sys.exit(1)
    else:
     #init_im=caffe.io_caffe.load_image(init_file)
     init_im = misc.imread(init_file)
     val=255
     #print init_im.shape
     (list_rows,list_cols)=np.where(init_im > val*0.075)
     
     orig_cords=np.zeros((len(list_rows),2))
     orig_cords[:,0]=list_rows
     orig_cords[:,1]=list_cols

     num_candidates=len(list_rows)
     print 'Number of candidates = '+str(num_candidates)

     os.remove(init_file)   


    print 'Computing Deep Features...'
    net.predict(input_data)
    #print 'Done'

    num_feats=5504
   
    feat_mat= gp.zeros((num_candidates,num_feats+1))



    start_ind=0

    #filters1 = net.params['conv1_1'][0].data
    num_filters=64
    feat1 = net.blobs['conv1_1'].data[start_ind, :num_filters]

    #filters2 = net.params['conv1_2'][0].data
    num_filters=64
    feat2 = net.blobs['conv1_2'].data[start_ind, :num_filters]
 
    #filters3 = net.params['conv2_1'][0].data
    num_filters=128                    
    feat3 = net.blobs['conv2_1'].data[start_ind, :num_filters]

    #filters4 = net.params['conv2_2'][0].data
    num_filters=128                  
    feat4 = net.blobs['conv2_2'].data[start_ind, :num_filters]

    
    num_filters=256  
    feat5 = net.blobs['conv3_1'].data[start_ind, :num_filters]


    #filters6 = net.params['conv3_2'][0].data
    num_filters=256
    feat6 = net.blobs['conv3_2'].data[start_ind, :num_filters]

    #filters7 = net.params['conv3_3'][0].data
    num_filters=256
    feat7 = net.blobs['conv3_3'].data[start_ind, :num_filters]

    #filters8 = net.params['conv3_4'][0].data
    num_filters=256                    
    feat8 = net.blobs['conv3_4'].data[start_ind, :num_filters]

    #filters9 = net.params['conv4_1'][0].data
    num_filters=512                  
    feat9 = net.blobs['conv4_1'].data[start_ind, :num_filters]

    #filters10 = net.params['conv4_2'][0].data
    num_filters=512  
    feat10 = net.blobs['conv4_2'].data[start_ind, :num_filters]


    #filters11 = net.params['conv4_3'][0].data
    num_filters=512                  
    feat11 = net.blobs['conv4_3'].data[start_ind, :num_filters]


    #filters12 = net.params['conv4_4'][0].data
    num_filters=512  
    feat12 = net.blobs['conv4_4'].data[start_ind, :num_filters]

    #filters13 = net.params['conv5_1'][0].data
    num_filters=512
    feat13 = net.blobs['conv5_1'].data[start_ind, :num_filters]


    #filters14 = net.params['conv5_2'][0].data
    num_filters=512
    feat14 = net.blobs['conv5_2'].data[start_ind, :num_filters]

    #filters15 = net.params['conv5_3'][0].data
    num_filters=512                    
    feat15 = net.blobs['conv5_3'].data[start_ind, :num_filters]

    #filters16 = net.params['conv5_4'][0].data
    num_filters=512                  
    feat16 = net.blobs['conv5_4'].data[start_ind, :num_filters]


    print 'Interpolating features...'


    feat_counter=1
    if True:
        
      if True:
          
         cur_feat=get_features(feat1,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]    
         
         cur_feat=get_features(feat2,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

          
                 
         cur_feat=get_features(feat3,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

         
                       
         cur_feat=get_features(feat4,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

         
                  
         cur_feat=get_features(feat5,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

          
                 
         cur_feat=get_features(feat6,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

         
               
         cur_feat=get_features(feat7,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]



  
         cur_feat=get_features(feat8,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]


         
          
         cur_feat=get_features(feat9,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

          
  
         cur_feat=get_features(feat10,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

          
           
         cur_feat=get_features(feat11,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]



         cur_feat=get_features(feat12,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]


         
           
         cur_feat=get_features(feat13,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]
 
         cur_feat=get_features(feat14,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]


       
         cur_feat=get_features(feat15,orig_cords,orig_dim)   
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]

          
                  
         cur_feat=get_features(feat16,orig_cords,orig_dim)
         feat_mat[:,feat_counter:feat_counter+cur_feat.shape[1]]=cur_feat
         feat_counter+=cur_feat.shape[1]   




    feats=feat_mat.as_numpy_array()

    feats=cm.CUDAMatrix(feats.T)


    print 'Predicting the boundaries...'



    mu=np.array(mu, order="F")
    mu=cm.CUDAMatrix(-1.0*mu)
    mu=mu.transpose()
    #mu=gp.garray(mu)

    sigma=np.array(sigma, order="F")
    sigma=cm.CUDAMatrix(1.0/sigma)
    sigma=sigma.transpose()
    #sigma=gp.garray(sigma)

   
    feats=feats.add_col_vec(mu)
    a=feats.mult_by_col(sigma)
    

    W1=hfl_data['W1']
    
    W1 = np.array(W1, order="F")
    #a = np.array(a, order="C")

    W2=hfl_data['W2'] 
    #W2 = np.array(W2, order="F")

    W3=hfl_data['W3'] 
    #W3 = np.array(W3, order="F")


    #a=cm.CUDAMatrix(feat_mat)
    b=cm.CUDAMatrix(W1.T)

    
    a1=cm.dot(a.transpose(),b)
    #a1=cm.dot(a,b)
    a1.mult(1.7159)
    a1.mult(2.0/3.0)

    a1=a1.asarray()



    a1=np.tanh(a1)
    a1=a1*0.5 # accounting for droput
    a1=np.c_[np.ones(num_candidates),a1] #adding bias


    a=cm.CUDAMatrix(a1.T)
    b=cm.CUDAMatrix(W2)


    a2=cm.dot(a.transpose(),b.transpose())
    a2.mult(1.7159)
    a2.mult(2.0/3.0)
    a2=a2.asarray()
    
    #a2=np.dot(a1,W2.T)

   
    a2=np.tanh(a2)
    a2=a2*0.5 # accounting for droput
    a2=np.c_[np.ones(num_candidates),a2] #adding bias


    a3=a2.dot(W3.T)


    #OUTPUT NORMALIZATION
    probs=get_probs(a3)


    HFL=np.zeros((num_rows,num_cols))
    for ii in range(len(list_rows)):
     cur_row=list_rows[ii]
     cur_col=list_cols[ii]
     
     cur_val=probs[ii,0]
     
     HFL[cur_row,cur_col]=cur_val*255

    #misc.imsave(output_dir+name+'.png',HFL)
    misc.imsave(output_file,HFL)


else:
   print 'ERROR: Invalid Image Extension'
   sys.exit(1)

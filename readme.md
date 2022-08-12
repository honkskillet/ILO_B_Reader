# ILO CXR interpretation
### Alexander White's Capstone
### UTHCT Occupaional Medicine 2023
----
## JOURNAL
#### 8/11
Worked mostly on rewritting code to use a different Keras Application (pretrain neural network).  Focused on the Xception network, the highest scoring Keras App. Below are network model.summary() statements. Most of the complexity is hidden within the pretrained layer.  Both layers were initially trained on the imagenet dataset.  I retrained both on a flower dataset with 5 types of flower.  I only trained for 2 epochs because it is VERY slow on my computer.  It took 30m13s to train the slower Xception network.  The benefit of Xception over MobileNetV2 is that it can take a 299 by 299 pixel image, while MNv2 can only take a 224 x 224.
```
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_7 (InputLayer)        [(None, 299, 299, 3)]     0         
                                                                 
 xception (Functional)       (None, 10, 10, 2048)      20861480  
                                                                 
 global_average_pooling2d_2   (None, 2048)             0         
 (GlobalAveragePooling2D)                                        
                                                                 
 dense_2 (Dense)             (None, 5)                 10245     
                                                                 
=================================================================
Total params: 20,871,725
Trainable params: 10,245
Non-trainable params: 20,861,480
_________________________________________________________________
```
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer_1 (KerasLayer)  (None, 1280)              2257984   
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 2,264,389
Trainable params: 6,405
Non-trainable params: 2,257,984
_________________________________________________________________
```

```
Epoch 1/2
86/86 [==============================] - 839s 9s/step - loss: 0.4555 - sparse_categorical_accuracy: 0.8536
Epoch 2/2
86/86 [==============================] - 877s 10s/step - loss: 0.3399 - sparse_categorical_accuracy: 0.8928
<keras.callbacks.History at 0x17cbe0ca0>
```
#### 8/10
- Working on Keras CNN with transfer learning
- First working CNN.  Uses a pretrained Keras application to solve know problem (identification of 5 different types of flowers from a data... roses, tulips, etc)
- Mr Reeves has found most of older-than 2008 studies.  Nickie provided some info for him to locate the 2 remaining.  Nickie has found most of the physical film studies; some might be missing the films.
#### 8/9
- In service exam!
- Afternoon board/exam review and meetings.
- Worked on code for transfer learning classification of NIH images (PA normals vs PA fibrosis)
- Lung Segmentation (Masking out everything but lung parenchyma) Research 
  - https://www.kaggle.com/datasets/farhanhaikhan/unet-lung-segmentation-weights-for-chest-x-rays/code
  - https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/notebook
#### 8/8
- Researching "transfer learning".  That is were you use parts of state of the art neural networks instead of trying to create an entire NN from scratch.  This are good for tasks such a image categorization, which is what I am doing here. Some good resources.
  - https://keras.io/api/applications/ (list of different networks, aka 'applications')
  - https://www.youtube.com/watch?v=LsdxvjLWkIY
  - https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/18_transfer_learning/cnn_transfer_learning.ipynb
  - https://github.com/nachi-hebbar/Transfer-Learning-Keras/blob/main/TransferLearning.ipynb
  - https://www.youtube.com/watch?v=lHM458ZsfkM
  - https://github.com/aladdinpersson/Machine-Learning-Collection
  - https://www.youtube.com/watch?v=WJZoywOG1cs
#### 8/5
- Interestingly, in NIH dataset there are 648 studies with fibrosis taken in the PA projection.  (Only 79 AP studies)  The PA studies would be more applicable to ILO CXR comparison because those are also PA projection studies.
```
Finding Labels  Edema  Fibrosis  No Finding
View Position                              
AP                577        79       21059
PA                 51       648       39302
```
More work with DataFrames
See read_csv.ipynb for more
#### 8/4
- Working with CSV (comma separated file) which identifies (see read_csv.py)
- Researched dataframes (ie Panda.py)
- Practicum presentations day (all lunch and afternoon)
#### 8/3
- Talked with Nickie regarding best ways to obtain pantex images
- Viewed stack of old x-ray films (pre mid-2008) in the H buildling filing room
- Met with Mr Ross (Rad tech) about obtaining Pantex dicoms directly from PAX
- Mr Ross informed me that radiology no longer has scanner for old x-ray films (physical films)
- Sent list of studies to Mr Ross to obtain (just the studies with + profusion scores).  Aprrox 1/3 of these studies are pre 2008
- Completed download of NIH images
#### 8/2
- Met with Dr Rowlett, planning
- Started NIH images download
----
## References
- Lung Segmentation from Chest X-ray Data: https://www.kaggle.com/code/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset/notebook
- Prediction of Pulmonary Fibrosis Based on X-Rays by Deep Neural Network:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8976624/  
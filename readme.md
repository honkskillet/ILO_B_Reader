# ILO CXR interpretation
### Alexander White's Capstone
### UTHCT Occupaional Medicine 2023
----
## JOURNAL
#### 8/10
Working on Keras CNN with transfer learning
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
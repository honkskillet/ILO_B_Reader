import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv('Data_Entry_2017_v2020.csv')

## Prints the first 5 records so we can see how the datafram is formatted
print(df.head())
print();
## Print all the diagnoses (aka findings)
print(df['Finding Labels'].value_counts());
## Print the number of PA & AP view studies in the data
print(df['View Position'].value_counts());
print();
## Find out how many studies have fibrosis and how many have ONLY fibrosis as finding

##Search for just the studies that have fibrosis or edema
filtered_df = df[df['Finding Labels'].str.fullmatch("Fibrosis|Edema|No Finding")]
print(filtered_df);
print(pd.crosstab(filtered_df['View Position'],filtered_df['Finding Labels']));
#print(df.describe()); 

img = mpimg.imread('dicom/NIH_images/00000006_000.png')
imgplot = plt.imshow(img, cmap='gray')
plt.show()
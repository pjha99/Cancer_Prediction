# Project Name
Cancer Detetction Project: 

### Project Intro/Objective
Cancer is the name given to a collection of related diseases. In all types of cancer, some of the body’s cells begin to divide without stopping and spread into surrounding tissues.

Cancer can start almost anywhere in the human body, which is made up of trillions of cells. Normally, human cells grow and divide to form new cells as the body needs them. When cells grow old or become damaged, they die, and new cells take their place.

When cancer develops, however, this orderly process breaks down. As cells become more and more abnormal, old or damaged cells survive when they should die, and new cells form when they are not needed. These extra cells can divide without stopping and may form growths called tumors.

In this Project, we will build and train a model using human cell records, and classify cells to whether the samples are benign or malignant. 


### Methods Used
1. Random forest
2. Hypertuning methods
    - Randomized Search CV
    - Grid Serach CV

### Technologies
- Python

### Data Set
<p> The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007).[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are: </p>

### Field name    
----------------------------------------
1. ID	Clump       
2. Clump	        
3. UnifSize	    
4. UnifShape	    
5. MargAdh	    
6. SingEpiSize	
7. BareNuc	    
8. BlandChrom	    
9. NormNucl	    
10. Mit	        
11. Class	        

Field BareNuc is a categorical feature, we will be using one hot encoding to convert it into the numerical Value.

## Exploratory Data Analysis

![Image](Images/Capture_cancer.png)



## Shape of Train and Test Data

- Train Data (546, 9) (546,)
- Test Data (137, 9) (137,)


## Getting Started

1. Raw Data is being kept under the project repo with the name drug200.csv    
2. Data processing/transformation scripts are coded under cancer_prediction.ipynb


## Featured Notebooks/Analysis/Deliverables
- cancer_prediction.ipynb


## Metrics Used to determine results:
1. Accuracy <br/>
2. Precision <br/>
3. Recall <br/>
4. F1-Score <br/>

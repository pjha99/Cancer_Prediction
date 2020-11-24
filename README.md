# Project Name
Cancer Detetction Project: 


#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro/Objective
In this Project, we will build and train a model using human cell records, and classify cells to whether the samples are benign or malignant. 


### Methods Used
*Random forest
*Hypertuning methods
    .Randomized Search CV
    .Grid Serach CV

### Technologies
*Python

## Project Description
The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are:

Field name	Description
ID	Clump thickness
Clump	Clump thickness
UnifSize	Uniformity of cell size
UnifShape	Uniformity of cell shape
MargAdh	Marginal adhesion
SingEpiSize	Single epithelial cell size
BareNuc	Bare nuclei
BlandChrom	Bland chromatin
NormNucl	Normal nucleoli
Mit	Mitoses
Class	Benign or malignant

Field BareNuc is a categorical feature, we will be using one hot encoding to convert it into the numerical Value.

## Shape of Train and Test Data

Train Data (546, 9) (546,)
Test Data (137, 9) (137,)


## Getting Started

1. Raw Data is being kept under the project repo with the name drug200.csv    
2. Data processing/transformation scripts are coded under cancer_prediction.ipynb


## Featured Notebooks/Analysis/Deliverables
*cancer_prediction.ipynb


## Metrics Used to determine results:
*Accuracy
*Precision 
*Recall
*F1-Score
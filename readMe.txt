-------------------------------------------------------

Source Code
-------------------------------------------------------

This package contains source code for submission: Sparse Maximum Margin Learning From Multimodal Recurrent States of Human Behavioral Patterns

Specification of dependencies：

The code is written in Matlab2021 with deep learning and parallel computing toolbox. Based on the specified software version, no third-party library is required for the environment setting. All files need to be on the same path in order for the sampling algorithms to work.

Datasets:

Stored in 'Data1' and 'Data2' folder, respectively.

Evaluation:

1. For dataset 1, run Evaluation1.m

2. For dataset 2, run Evaluation2.m

Demo Video:

3. A demo video with screen recording is provided to illustrate how to run the evaluation code for dataset 1 in Matlab2021. At the end of demo, the prediction accuracy, along with the prediction and ground truth class for data instances 1-15, is printed on the command window. 


Training:

Train1.m
The method that implements the proposed data fusion algorithm for fusing topic and gaze patterns.

Train2.m
The method that implements the proposed data fusion algorithm for fusing path, gaze and PMD patterns.

The following functions are called by the main methods.

FuncInter.m
The function that generates intermediate files to store information of gaze and path.

FuncProc.m
The function that processes gaze and path data to align the centers and facilitate calculation.

FuncCalc.m
The function that computes the probability of pattern assignments.




# Motion Detection
In this project, we try to detect the motion of an object in an image using Lucas Kanade method. We compute dense flow field vector using Hierarchial Luca & Kanade. The first part of this project deals with Optical flow using Lucas Kanade algorithm. We find that for for large shifts, lucas & Kanade fails to record the movement values accurately. In order to ovecome this limitation, we used Hierarchial Lucas-Kanade method. Later, we used this implementation for Frame interpolations as well.

## Instructions to run the code:
	1) run experiment.py
	
## Outputs
##Small shifts demo:

![state1](input_images/TestSeq/Shift0.png)  ![state2](input_images/TestSeq/ShiftR2.png)

![Motion_vector](output/ps4-1-a-1.png)

## Large Shifts Demo:
![state1](input_images/Urban2/urban01.png)  ![state2](input_images/Urban2/urban02.png)

![Motion_vector](output/ps4-4-b-1.png)

## Frame Interpolation:
Interpolation of intermediate frames from 3 input frames:

![state1](input_images/MiniCooper/mc01.png)  ![state2](input_images/MiniCooper/mc02.png) ![state2](input_images/MiniCooper/mc03.png)

![Motion_vector](output/output/output.gif)

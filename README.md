# Reconstuct 3D information using stereo system with OpenCV
This repository contains the script which can reconstruct the 3D information through a stereo vision system.
These scripts are designed for the experiment setup using in [Nordlinglab QPD project](https://www.nordlinglab.org/quantpd/)

## The stereo system and triangulation algorithm
The triangulation algorithm is a classic computer vision algorithm based on a stereo vision system.
If you are interested in the theory, using these keywords might help you to understand the scripts.


*Keywords: Stereo vision, Epipolar geometry, Camera matrix, Camera calibartion, Zhang's camera calibration algorithm, Triangulation*

In summary, reconsturt 3D information require:
1. A stereo vision system (two camera)
1. Camera matrix of these two cameras
1. A pair of corresponding points in both camera views

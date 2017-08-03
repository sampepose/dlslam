August 3, 2017

first ran srikanth's rectification code in /stereo to get two rectified images. there should therefore be NO rotation, only translation.

predicted relative transformation matrix of camera pose between the two rectified images using demon:

[[  9.99975686e-01  -4.65409052e-04  -6.95785515e-03  -9.73569900e-02]
 [  4.22427894e-04   9.99980830e-01  -6.17755126e-03   1.21298246e-02
 [  6.96059685e-03   6.17446186e-03   9.99956712e-01   1.90458749e-03]
 [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]

If we scale the translation by 370, we get a translation of:
[-36.0220863 ,   4.4880351 ,   0.70469737]

we know from the rectification that ground truth is no rotation and translation of <-36.58, 0, 0>,
so our prediction is pretty close! (disregarding units/scaling)

We predict a slight rotation even thought gt is no rotation.

The predicted depth map is decent. We may be able to get an even better depth if we can better predict camera motion. we can predict better camera motion by finetuning or changing the architecture.

Next step is to re-run our initial experiment and see if rectified images do better. It is likely, as we don't take into account the distortion intrinsics without this rectification.

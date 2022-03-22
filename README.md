# stereo-triangulation
Find the 3-D position of objects using two cameras. The first part, following [Machicoane _et al._](https://aip.scitation.org/doi/full/10.1063/1.5080743) [**1**], enables the mapping of camera pixel locations to optical paths in 3-D space, given a set of calibration images that can be obtained by moving a planar calibration target along an axis. The second part enables the "matching" of objects between two 2-D views, yielding their 3-D position, by finding the near-intersection of the optical paths corresponding to the detected object in each view.

This implementation approximates the optical paths as straight lines, so it cannot be used for systems in which there are changes in the index of refraction within the measurement volume. (Changes in the index of refraction between the measurement volume and the cameras, however, are not an issue.)

## Sample usage

```python
from stereo.camera import camera
from stereo.stereo_system import StereoSystem, Matcher

# load camera calibration information that can be created using the calibration GUI
calib_A = camera.calibration_from_dict('camera_A.pkl')
calib_B = camera.calibration_from_dict('camera_B.pkl')

# create a "stereo system" comprised of the two cameras
ss = StereoSystem(calib_A,calib_B)

# load DataFrames giving the location and size of 2-D objects you've detected in each view
df_A, df_B = your_function_to_load_your_2D_data()

# match detected objects
m = Matcher(df_A,df_B,ss)
df_3d = Matcher.match()
```

## Assumed coordinate system

The calibration procedure described in [1] involves taking images of a calibration plate as it is traversed through the measurement domain. Here, we define the plate to be parallel to the _x-z_ plane, so the orthogonal direction in which it is traversed is the _y_ direction. The points marked on the calibration plate should lie on rows corresponding to the _x_ direction and columns corresponding to the _z_ direction.

## Calibration GUI

After obtaining images of a calibration target obtained following [1], save them as `.tif` files and place them in directories `BASE_FOLDER\CAMERA_NAME_1\` and `BASE_FOLDER\CAMERA_NAME_2\`, each named `y_Ymm.tif`, where `Y` is the _y_-position of the plate in that image, in mm.

Then, call the calibration GUI in [`stereo/camera/calibration_gui.py`](stereo/camera/calibration_gui.py) once for each camera. The GUI facilitates the construction of the calibration files for each camera that are used in initializing a `StereoSystem`.

## Installation

Download the code to your computer, then run `pip install -e .` .

## References

**[1]** [Machicoane, N., Aliseda, A., Volk, R., & Bourgoin, M. (2019). A simplified and versatile calibration method for multi-camera optical systems in 3D particle imaging. _Review of Scientific Instruments_, 90(3), 035112.](https://aip.scitation.org/doi/full/10.1063/1.5080743)

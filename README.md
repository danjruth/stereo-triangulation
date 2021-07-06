# stereo-triangulation
Find the 3-D position of objects using two cameras. The first part, following [Machicoane _et al._](https://aip.scitation.org/doi/full/10.1063/1.5080743) [**1**], enables the mapping of camera pixel locations optical paths in 3-D space, given a set of calibration images that can be obtained by moving a planar calibration target along an axis. The second part enables the "matching" of objects between two 2-D views, yielding their 3-D position, by finding the near-intersection of the optical paths corresponding to the detected object in each view.

## Sample usage

```python
from stereo.camera import camera
from stereo.stereo_system import StereoSystem, Matcher

# load camera calibration information that can be created using the calibration GUI
calib_A = camera.calibration_from_dict('calibration_A.pkl')
calib_B = camera.calibration_from_dict('calibration_B.pkl')

# create a "stereo system" comprised of the two cameras
ss = StereoSystem(calib_A,calib_B)

# load DataFrames giving the location and size of 2-D objects you've detected in each view
df_A, df_B = your_function_to_load_your_2D_data()

# match detected objects
m = Matcher(df_A,df_B,ss)
df_3d = Matcher.match()
```

## Installation

Download the code to your computer, then run `pip install -e .` .

## References

**[1]** [Machicoane, N., Aliseda, A., Volk, R., & Bourgoin, M. (2019). A simplified and versatile calibration method for multi-camera optical systems in 3D particle imaging. _Review of Scientific Instruments_, 90(3), 035112.](https://aip.scitation.org/doi/full/10.1063/1.5080743)

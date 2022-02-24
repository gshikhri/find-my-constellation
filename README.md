# find-my-constellation: Map an image of night sky and find any constellations within it

- [Repository overview](#repository-overview)
- [Issues and questions](#issues-and-questions)

Find the corresponding stars (point sources) and match a constellation using Fourier based cross-correlation. The repository uses the registration usitility within [skimage.registration.phase_cross_correlation](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation) for obtaining the tranformation parameters between the input image and a reference image (in future generated using (SkyCoord from atropy.io)[https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html])

## Repository overview

The basic principle behind mapping a pattern of point sources to a reference image can be broken down into two steps:
1. Identify the projection transformation between the two images. There are many ways to do this, here I am using Fourier based registration methods as they are quite fast since all the computation is done in the fourier domain (Convolution in spatial domain is multiplication in fourier domain).
![](https://github.com/gshikhri/find-my-constellation/blob/main/assets/Step1.png)
2. Map the point sources in the both images after warping the input image to the reference image and identify stars common within the two images. One way to do this is to use [RANSAC](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.ransac) but the time complexity of RANSAC increases with increase in the number of features (point sources)
Instead, here I use an approach that uses [k-d-Trees](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) to find nearest neighbors between the stars in two images. The result is that the stars within the constellation can be correctly identified while the false positives are discarded. 
![](https://github.com/gshikhri/find-my-constellation/blob/main/assets/Step2.png)


## Issues and questions
Find-my-constellation is a work in progress right now. The immediate goal is to transition from Jupyter-notebooks to an interactive dashboard. In case you need help or have suggestions or you want to report an issue, please do so in a reproducible example at the corresponding [GitHub page](https://github.com/gshikhri/find-my-constellation/issues).

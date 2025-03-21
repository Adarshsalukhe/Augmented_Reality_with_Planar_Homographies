# Augmented_Reality_with_Planar_Homographies

Homography is a fundamental concept in computer vision that describes the relationship between two different perspectives of a planar surface. It allows us to map points from one image to another, as long as the images capture the same scene from different viewpoints. Homographies are widely used in applications such as image stitching, 3D reconstruction, and augmented reality.

Homography is a transformation matrix that defines a projective transformation between two images. It is a 3x3 matrix that relates the pixel coordinates of one image to the pixel coordinates of another image. If the two images depict the same scene from different viewpoints, and the scene lies on a plane, homography can be used to warp one image onto another.

Image Stiching for Panaroma - Aligns images to create a panoramic view by warping them using homography.
Augmented Reality: Projects virtual objects onto real-world surfaces by estimating the homography between the camera view and the real-world plane.
Planar Object Detection: Identifies and tracks objects by estimating their homography in different images or video frames.
3D Reconstruction: Helps recover 3D geometry from multiple images.

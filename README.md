# Unsupervised Point Cloud Representation Learning by Clustering and Neural Rendering
This is the part of official code for IJCV review
# Abstract
Data augmentation has contributed to the rapid advancement of unsupervised learning on 3D point clouds. However, we argue that data augmentation is not ideal, as it requires a careful application-dependent selection of the types of augmentations to be performed, thus potentially biasing the information learned by the network during self-training.
Moreover, several unsupervised methods only focus on unimodal information, thus potentially introducing challenges in the case of sparse point clouds.
To address these issues, we propose an augmentation-free unsupervised approach for point clouds, named CluRender, to learn transferable point-level features by leveraging unimodal information for soft clustering and cross-modal information for neural rendering.
Soft clustering enables self-training through a pseudo-label prediction task, where the affiliation of points to their clusters is used as a proxy under the constraint that these pseudo-labels divide the point cloud into approximate equal partitions.
This allows us to formulate a clustering loss to minimize the standard cross-entropy between pseudo and predicted labels.
Neural rendering generates photorealistic renderings from various viewpoints to transfer photometric cues from 2D images to the features. 
The consistency between rendered and real images is then measured to form a fitting loss, combined with the cross-entropy loss to self-train networks.
Experiments on downstream applications, including 3D object detection, semantic segmentation, classification, part segmentation, and few-shot learning, demonstrate the effectiveness of our framework in outperforming state-of-the-art techniques.
# 
>>>>>>> clurender

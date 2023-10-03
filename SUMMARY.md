**LVIS: A Dataset for Large Vocabulary Instance Segmentation v1.0** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is applicable or relevant across various domains. 

The dataset consists of 159623 images with 3474084 labeled objects belonging to 1203 different classes including *tennis_racket*, *short_pants*, *ski_pole*, and other: *sock*, *skateboard*, *bench*, *umbrella*, *chair*, *pillow*, *person*, *dog*, *sunglasses*, *sofa*, *belt*, *jean*, *drawer*, *ski*, *spectacles*, *backpack*, *necklace*, *necktie*, *horse*, *trousers*, *wheel*, *flag*, *hat*, *plate*, *shoe*, and 1175 more.

Images in the LVIS dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 40609 (25% of the total) unlabeled images (i.e. without annotations). There are 4 splits in the dataset: *test challenge* (19822 images), *test dev* (19822 images), *training set* (100170 images), and *validation set* (19809 images). The dataset was released in 2019 by the Facebook AI Research (FAIR).

<img src="https://github.com/dataset-ninja/lvis/raw/main/visualizations/poster.png">

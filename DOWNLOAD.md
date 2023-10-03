Dataset **LVIS** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/o/M/iG/UG1cNN0qpiEREfCWDfeUFUTLZIZrhnowclhQTC9CsvvoBjiwzs5t2Ip3CfroRV4pOY0G7QTrGbYI8f7vbvp5ha0mA6qGeUIOPIBOmybXBw7trPK16q114ZNeYhvK.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='LVIS', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be downloaded here:

- [Training set 1,270,141 instances (1 GB)](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip)
- [Training set 100,170 images (18 GB)](http://images.cocodataset.org/zips/train2017.zip)
- [Validation set 244,707 instances (192 MB)](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip)
- [Validation set 19,809 images (1 GB)](http://images.cocodataset.org/zips/val2017.zip)
- [Test Dev info (4 MB)](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_image_info_test_dev.json.zip)
- [Test Dev 19,822 images (6 GB)](http://images.cocodataset.org/zips/test2017.zip)
- [Test Challenge  info (4 MB)](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_image_info_test_challenge.json.zip)
- [Test Challenge  19,822 images (6 GB)](http://images.cocodataset.org/zips/test2017.zip)

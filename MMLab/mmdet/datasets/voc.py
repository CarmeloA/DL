from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDataset(XMLDataset):

    CLASSES = ( 'bigknife', 'bottle','cellphone','cup','keys','knife','laptop','lighter',
            'pliers','screwdriver','smallknife', 'spoon&fork','umbrella','unnormalknife','watch'
                )
    # CLASSES = ('bigknife', 'cup', 'knife', 
    #            'pliers', 'screwdriver', 'smallknife', 'spoon_fork', 'umbrella', 'unnormalknife', 
    #            )

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        elif 'VOC2020' in self.img_prefix:
            self.year = 2020
        elif 'VOC2021' in self.img_prefix:
            self.year = 2021
        elif 'VOC2022' in self.img_prefix:
            self.year = 2022
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')

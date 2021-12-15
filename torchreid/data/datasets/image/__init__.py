from __future__ import absolute_import, print_function

from .classification import Classification, ClassificationImageFolder, ExternalDatasetWrapper, MultiLabelClassification
from .compcars import CompCars
from .cuhk01 import CUHK01
from .cuhk02 import CUHK02
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .globalme import (InternalAirport, InternalCameraTampering,
                       InternalGlobalMe, InternalMall, InternalPSVIndoor,
                       InternalPSVOutdoor, InternalSSPlatform,
                       InternalSSStreet, InternalSSTicket, InternalWildtrack,
                       MarketTrainOnly)
from .grid import GRID
from .ilids import iLIDS
from .lfw import LFW
from .market1501 import Market1501
from .msmt17 import MSMT17
from .prid import PRID
from .sensereid import SenseReID
from .universe_models import UniverseModels
from .vehicle1m import Vehicle1M
from .veriwild import VeRiWild
from .vgg_face2 import VGGFace2
from .viper import VIPeR
from .vmmrdb import VMMRdb
from .vric import VRIC

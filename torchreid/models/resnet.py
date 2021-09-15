from torchreid.ops.dropout import Dropout
import timm
from .common import ModelInterface
from torchreid.ops import Dropout
import timm

__ALL__ = ['resnet101D', 'resnet101', 'wide_resnet101', 'resnext101']

class ResnetTimm(ModelInterface):
    def __init__(self,
                model_name,
                pretrained=False,
                dropout_cls = None,
                num_classes=1000,
                pooling_type='avg',
                **kwargs):
        super().__init__(**kwargs)
        dropout_rate = dropout_cls.p
        self.pooling_type = pooling_type
        self.dropout = Dropout(**dropout_cls)
        self.model = timm.create_model(model_name,
                                        pretrained=pretrained,
                                        num_classes=num_classes,
                                        drop_rate=dropout_rate)
        self.num_features = self.model.num_features
        assert self.loss in ['softmax', 'asl'], "Resnet supports only softmax or ASL losses"

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        assert not get_embeddings
        y = self.model.forward_features(x)
        if return_featuremaps:
            return y
        glob = self.model.global_pool(y)
        if self.model.drop_rate:
            glob = self.dropout(glob)
        logits = self.model.fc(glob)

        if not self.training and self.is_classification():
            return [logits]

        elif self.loss in ['softmax', 'am_softmax', 'asl']:
                out_data = [logits]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return tuple(out_data)

def resnet101D(pretrained=False, **kwargs):
    """
    Constructs a resnet101d model
    """
    net = ResnetTimm('resnet101d', pretrained=pretrained, **kwargs)
    return net

def resnet101(pretrained=False, **kwargs):
    """
    Constructs a resnet101 (torchvision weights) model
    """
    net = ResnetTimm('tv_resnet101', pretrained=pretrained, **kwargs)
    return net

def wide_resnet101(pretrained=False, **kwargs):
    """
    Constructs a wide_resnet101_2  model
    """
    net = ResnetTimm('wide_resnet101_2', pretrained=pretrained, **kwargs)
    return net

def resnext101(pretrained=False, **kwargs):
    """
    Constructs a resnext101_32x4d  model
    """
    net = ResnetTimm('resnext101_32x4d', pretrained=pretrained, **kwargs)
    return net

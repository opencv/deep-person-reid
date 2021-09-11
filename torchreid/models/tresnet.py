from .common import ModelInterface
import timm

__ALL__ = ['tresnet']

class TResnetTimm(ModelInterface):
    def __init__(self,
                pretrained=False,
                dropout_cls = None,
                lr_finder=None,
                num_classes=1000,
                num_features=2432,
                **kwargs):
        super().__init__(**kwargs)
        dropout_rate = dropout_cls.p
        self.lr_finder = lr_finder
        self.num_features = num_features
        self.model = timm.create_model('tresnet_m',
                                        pretrained=pretrained,
                                        num_classes=num_classes,
                                        drop_rate=dropout_rate)
        assert self.loss in ['softmax', 'asl'], "TResnetTimm supports only softmax or ASL losses"

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        assert not get_embeddings
        y = self.model.forward_features(x)
        if return_featuremaps:
            return y

        logits = self.model.head(y)

        if not self.training and self.is_classification():
            return [logits]

        elif self.loss in ['softmax', 'am_softmax', 'asl']:
            if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
                out_data = logits
            else:
                out_data = [logits]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        if self.lr_finder.enable and self.lr_finder.lr_find_mode == 'automatic':
            return out_data
        return tuple(out_data)

def tresnet(pretrained=False, **kwargs):
    """
    Constructs a TResnetM model
    """
    try:
        import inplace_abn
    except ImportError:
        print("No module 'inplace_abn' found. To use TResNet you need to install 'optional-requirments.txt'")
        exit()
    net = TResnetTimm(pretrained=pretrained, **kwargs)
    return net

from naive_cnn import *
from vgg16 import *
from vgg16_top import *
from always_female import *
from face_gender import *
from ensemble import *
from random import *

def load_model(name, type, weights_dir, checkpoint=None):
    """Load a model

    Args:
        name (str): Arbitrary name for model
        type (str): Type of model
        weights_dir (str): Where to store weights
        checkpoint (str): (optional) Path to HDF5 checkpoint
    """
    if type == 'always_female':
        model = AlwaysFemale(weights_dir, name)
    elif type == 'random':
        model = Random(weights_dir, name)
    elif type == 'naive_cnn':
        model = NaiveCNN(weights_dir, name)
    elif type == 'vgg16':
        model = VGG16(weights_dir, name)
    elif type == 'face_gender':
        model = FaceGender('./models/pretrained/haarcascade_frontalface_default.xml',
                           './models/pretrained/face_gender_cnn.h5', weights_dir, name)
    elif type == 'ensemble':
        vgg_model = load_model('vgg16_'+name, 'vgg16', weights_dir, './results/vgg16_amazon.h5')
        face_gender_model = load_model('face_gender_'+name, 'face_gender', weights_dir, None)
        model = Ensemble([vgg_model, face_gender_model], weights_dir, name)
    else:
        raise NotImplementedError('Model '+type+' not implemented!')
    if checkpoint != None:
        model.load(checkpoint)
    return model

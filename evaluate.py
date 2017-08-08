import sys
from helpers import *
from models import NaiveCNN, VGG16, AlwaysFemale, FaceGender, Ensemble

# Inputs
name = sys.argv[1]
model_type = sys.argv[2]
data_dir = sys.argv[3]
checkpoint = sys.argv[4] if len(sys.argv) > 4 else None

# Predict
if model_type == 'naive_cnn':
    model = NaiveCNN(tmp_dir, name)
elif model_type == 'vgg16':
    model = VGG16(tmp_dir, name)
elif model_type == 'always_female':
    model = AlwaysFemale(tmp_dir, name)
elif model_type == 'face_gender':
    model = FaceGender('./models/pretrained/haarcascade_frontalface_default.xml', './models/pretrained/face_gender_cnn.h5', tmp_dir, name)
elif model_type == 'ensemble':
    vgg_model = VGG16(tmp_dir, 'vgg16_'+name)
    vgg_model.load('./results/vgg16_data_v2.h5')
    face_gender_model = FaceGender('./models/pretrained/haarcascade_frontalface_default.xml', './models/pretrained/face_gender_cnn.h5', tmp_dir, 'face_gender_'+name)
    model = Ensemble([vgg_model, face_gender_model], tmp_dir, name)
else:
    raise NotImplementedError('Model '+model_type+' not implemented!')
model.load(checkpoint)
img_dir, meta_dir, train_dir, test_dir = get_directories(data_dir)
predictions = model.predict(test_dir)

# Evaluate
evaluate(name, predictions, test_dir)

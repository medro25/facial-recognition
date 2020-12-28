from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
listN=[]
listA=[]
listD=[]
listH=[]
listSad=[]
listSurprise=[]
listF=[]


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        if(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Neutral'):
            listN.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Happy'):
            listH.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Angry'):
            listA.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Surprise'):
            listSurprise.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Sad'):
            listSad.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Fear'):
            listF.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        elif(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]=='Disgust'):
            listD.append(FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)])
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

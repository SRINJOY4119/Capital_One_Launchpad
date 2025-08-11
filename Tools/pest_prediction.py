import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob

class PestPredictor:
    def __init__(self, model_path='../models/Pest_prediction/pests.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            'aphids', 'army_worm', 'beetle', 'bollworm', 'earthworm', 
            'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer'
        ]
        print("Model loaded successfully!")
    
    def predict_single_image(self, image_path):
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            predicted_pest = self.class_names[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            return predicted_pest, confidence
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    

def main():
    predictor = PestPredictor()

    image_path1 = "../Dataset/pest/test/aphids/jpg_0 - Copy.jpg"
    image_path2 = "../Dataset/pest/test/aphids/jpg_2 - Copy.jpg"
    pest1, confidence1 = predictor.predict_single_image(image_path1)
    print(f"Image: {os.path.basename(image_path1)} - Predicted Pest: {pest1} (Confidence: {confidence1:.1%})")
    pest2, confidence2 = predictor.predict_single_image(image_path2)
    print(f"Image: {os.path.basename(image_path2)} - Predicted Pest: {pest2} (Confidence: {confidence2:.1%})")


if __name__ == "__main__":
    main()

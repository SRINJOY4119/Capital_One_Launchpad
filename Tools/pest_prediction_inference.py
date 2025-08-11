import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2
import pickle
from pathlib import Path

class PestPredictionInference:
    def __init__(self, model_path='../models/Pest_prediction/pests.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'aphids', 'army_worm', 'beetle', 'bollworm', 'earthworm', 
            'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained pest prediction model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ Pest prediction model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Available pest classes: {self.class_names}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        try:
            # Load and resize image
            img = image.load_img(image_path, target_size=(224, 224))
            
            # Convert to array
            img_array = image.img_to_array(img)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def predict_pest(self, image_path):
        """
        Predict pest type from image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Prediction results with pest type and confidence
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return {'success': False, 'error': f'Image file not found: {image_path}'}
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class
            predicted_index = np.argmax(predictions[0])
            predicted_pest = self.class_names[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_indices:
                pest_name = self.class_names[idx]
                pest_confidence = float(predictions[0][idx])
                top_3_predictions.append((pest_name, pest_confidence))
            
            return {
                'success': True,
                'predicted_pest': predicted_pest,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions,
                'image_path': image_path,
                'all_probabilities': {self.class_names[i]: float(predictions[0][i]) 
                                    for i in range(len(self.class_names))}
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}
    
    def predict_batch(self, image_paths):
        """Predict pests for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict_pest(img_path)
            results.append(result)
        return results
    
    def get_pest_info(self, pest_name):
        """Get information about a specific pest"""
        pest_info = {
            'aphids': {
                'description': 'Small, soft-bodied insects that feed on plant sap',
                'damage': 'Stunted growth, yellowing leaves, honeydew secretion',
                'control': 'Insecticidal soap, neem oil, beneficial insects'
            },
            'army_worm': {
                'description': 'Caterpillars that move in groups across crops',
                'damage': 'Defoliation, stem cutting, crop destruction',
                'control': 'Bt sprays, pheromone traps, biological control'
            },
            'beetle': {
                'description': 'Hard-bodied insects with chewing mouthparts',
                'damage': 'Leaf holes, root damage, fruit scarring',
                'control': 'Row covers, beneficial nematodes, targeted insecticides'
            },
            'bollworm': {
                'description': 'Moth larvae that attack cotton bolls and other crops',
                'damage': 'Boll damage, yield reduction, quality loss',
                'control': 'Bt cotton, pheromone traps, IPM strategies'
            },
            'earthworm': {
                'description': 'Beneficial soil organisms (not typically a pest)',
                'damage': 'Generally beneficial - improves soil structure',
                'control': 'Usually not needed - consider beneficial'
            },
            'grasshopper': {
                'description': 'Jumping insects with strong hind legs',
                'damage': 'Defoliation, crop destruction during outbreaks',
                'control': 'Biological agents, barrier crops, targeted spraying'
            },
            'mites': {
                'description': 'Tiny arachnids that feed on plant cells',
                'damage': 'Stippling, webbing, leaf bronzing, reduced vigor',
                'control': 'Predatory mites, miticides, proper irrigation'
            },
            'mosquito': {
                'description': 'Flying insects (adults not typically crop pests)',
                'damage': 'Disease transmission, minor plant damage',
                'control': 'Water management, biological control, repellents'
            },
            'sawfly': {
                'description': 'Wasp-like insects with leaf-feeding larvae',
                'damage': 'Defoliation, skeletonized leaves, growth reduction',
                'control': 'Pruning, beneficial insects, selective insecticides'
            },
            'stem_borer': {
                'description': 'Larvae that bore into plant stems',
                'damage': 'Stem damage, wilting, plant death, yield loss',
                'control': 'Resistant varieties, biological control, stem treatment'
            }
        }
        return pest_info.get(pest_name, {'description': 'Information not available'})

def interactive_pest_prediction():
    """Interactive function for pest prediction"""
    try:
        predictor = PestPredictionInference()
        
        print("\n" + "="*60)
        print("🐛 PEST PREDICTION SYSTEM")
        print("="*60)
        
        # Get image path from user
        image_path = input("Enter the path to the pest image: ").strip()
        
        # Validate and predict
        result = predictor.predict_pest(image_path)
        
        print("\n" + "="*60)
        print("🔍 PEST PREDICTION RESULTS")
        print("="*60)
        
        if result['success']:
            print(f"✅ Predicted Pest: {result['predicted_pest'].upper()}")
            print(f"🎯 Confidence: {result['confidence']:.2%}")
            
            print("\n📊 Top 3 Predictions:")
            for i, (pest, conf) in enumerate(result['top_3_predictions'], 1):
                print(f"  {i}. {pest}: {conf*100:.2f}%")
            
            print(f"\n📁 Image: {result['image_path']}")
            
            # Get pest information
            pest_info = predictor.get_pest_info(result['predicted_pest'])
            print(f"\n📖 Pest Information:")
            print(f"  Description: {pest_info['description']}")
            print(f"  Damage: {pest_info['damage']}")
            print(f"  Control: {pest_info['control']}")
            
        else:
            print(f"❌ Error: {result['error']}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def batch_pest_prediction():
    """Batch prediction for multiple images"""
    try:
        predictor = PestPredictionInference()
        
        print("\n🐛 Batch Pest Prediction")
        print("-" * 40)
        
        # Get directory path
        directory = input("Enter directory path containing pest images: ").strip()
        
        if not os.path.exists(directory):
            print("❌ Directory not found!")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory, f"*{ext}")
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("❌ No image files found in directory!")
            return
        
        print(f"📁 Found {len(image_files)} images")
        
        # Process images
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {img_path.name}")
            result = predictor.predict_pest(str(img_path))
            results.append((img_path.name, result))
        
        # Display results
        print("\n" + "="*80)
        print("📊 BATCH PREDICTION RESULTS")
        print("="*80)
        
        successful = 0
        for filename, result in results:
            if result['success']:
                successful += 1
                print(f"✅ {filename}: {result['predicted_pest']} ({result['confidence']:.1%})")
            else:
                print(f"❌ {filename}: {result['error']}")
        
        print(f"\n📈 Summary: {successful}/{len(results)} successful predictions")
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")

def demo_pest_prediction():
    """Demo with sample predictions"""
    predictor = PestPredictionInference()
    
    # Sample image paths (you can modify these)
    sample_images = [
        "../Dataset/pest/train/aphids/jpg_0.jpg",
        "../Dataset/pest/train/beetle/jpg_1.jpg", 
        "../Dataset/pest/train/stem_borer/jpg_1.jpg"
    ]
    
    print("\n🐛 Demo Pest Predictions:")
    print("="*50)
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            result = predictor.predict_pest(img_path)
            if result['success']:
                print(f"\n📁 Image: {os.path.basename(img_path)}")
                print(f"🔍 Predicted: {result['predicted_pest']}")
                print(f"🎯 Confidence: {result['confidence']:.2%}")
                print(f"📊 Top 3: {', '.join([f'{p}({c:.1%})' for p, c in result['top_3_predictions']])}")
            else:
                print(f"❌ Error processing {img_path}: {result['error']}")
        else:
            print(f"⚠️ Sample image not found: {img_path}")

if __name__ == "__main__":
    print("🐛 Pest Prediction Tool")
    print("Choose an option:")
    print("1. Interactive single image prediction")
    print("2. Batch prediction for directory")
    print("3. Run demo with sample images")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            interactive_pest_prediction()
        elif choice == "2":
            batch_pest_prediction()
        elif choice == "3":
            demo_pest_prediction()
        else:
            print("Invalid choice. Running interactive prediction...")
            interactive_pest_prediction()
            
    except Exception as e:
        print(f"Error: {str(e)}")

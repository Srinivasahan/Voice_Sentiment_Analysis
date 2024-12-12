# from flask import Flask, jsonify
# import librosa
# import soundfile
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# import glob
# import os
# import sounddevice as sd
# from flask_cors import CORS
# import time

# app = Flask(__name__)
# CORS(app)

# # Define the emotions dictionary
# emotions = {
#     '01': 'neutral',
#     '02': 'calm',
#     '03': 'happy',
#     '04': 'sad',
#     '05': 'angry',
#     '06': 'fearful',
#     '07': 'disgust',
#     '08': 'surprised'
# }

# # Observed emotions
# observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# # Function to add noise for data augmentation
# def add_noise(data, noise_factor=0.005):
#     noise = np.random.randn(len(data))
#     augmented_data = data + noise_factor * noise
#     return augmented_data.astype(type(data[0]))

# # Function to extract features from audio
# def extract_feature(file_name=None, audio=None, sample_rate=22050, mfcc=True, chroma=True, mel=True):
#     if file_name:
#         with soundfile.SoundFile(file_name) as sound_file:
#             X = sound_file.read(dtype="float32")
#             sample_rate = sound_file.samplerate
#     else:
#         X = audio

#     result = np.array([])
#     if mfcc:
#         mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#         result = np.hstack((result, mfccs))
#     if chroma:
#         stft = np.abs(librosa.stft(X))
#         chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, chroma_features))
#     if mel:
#         mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, mel_features))

#     return result

# # Load data and split into training and testing sets
# def load_data(test_size=0.2, augment=False):
#     x, y = [], []
#     for file in glob.glob("/Users/sonika.n/PROJECT/ser/api/speech-emotion-recognition-ravdess-data/**/*.wav", recursive=True):
#         file_name = os.path.basename(file)
#         emotion = emotions.get(file_name.split("-")[2])
#         if emotion not in observed_emotions:
#             continue
#         feature = extract_feature(file_name=file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#         y.append(emotion)
#         if augment:
#             augmented_feature = add_noise(feature)
#             x.append(augmented_feature)
#             y.append(emotion)

#     if not x:
#         raise ValueError("No valid audio files found. Please check the dataset path and file format.")
    
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# # Train model on dataset
# x_train, x_test, y_train, y_test = load_data(test_size=0.25, augment=True)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # Initialize and train the model
# model = MLPClassifier(alpha=0.001, batch_size=64, epsilon=1e-08, hidden_layer_sizes=(256, 128, 64), learning_rate='adaptive', max_iter=800)
# model.fit(x_train, y_train)

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     duration = 5
#     sample_rate = 22050
#     print("Recording...")

#     # Record live audio on button press
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()

#     # Extract features and scale
#     feature = extract_feature(audio=audio, sample_rate=sample_rate, mfcc=True, chroma=True, mel=True)
#     feature = scaler.transform(np.expand_dims(feature, axis=0))

#     # Predict emotion
#     prediction = model.predict(feature)
#     predicted_emotion = prediction[0]

#     # Save the audio recording
#     audio_file_name = f"recording_{int(time.time())}.wav"
#     soundfile.write(audio_file_name, audio, sample_rate)

#     return jsonify({'predicted_emotion': predicted_emotion, 'audio_file': audio_file_name})

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, jsonify
# import librosa
# import soundfile
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# import glob
# import os
# import sounddevice as sd
# from flask_cors import CORS
# import time

# app = Flask(__name__)
# CORS(app)

# # Define the emotions dictionary
# emotions = {
#     '01': 'neutral',
#     '02': 'calm',
#     '03': 'happy',
#     '04': 'sad',
#     '05': 'angry',
#     '06': 'fearful',
#     '07': 'disgust',
#     '08': 'surprised'
# }

# # Observed emotions
# observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# # Function to add noise for data augmentation
# def add_noise(data, noise_factor=0.005):
#     noise = np.random.randn(len(data))
#     augmented_data = data + noise_factor * noise
#     return augmented_data.astype(type(data[0]))

# # Function to extract features from audio
# def extract_feature(file_name=None, audio=None, sample_rate=22050, mfcc=True, chroma=True, mel=True):
#     if file_name:
#         with soundfile.SoundFile(file_name) as sound_file:
#             X = sound_file.read(dtype="float32")
#             sample_rate = sound_file.samplerate
#     else:
#         X = audio

#     result = np.array([])
#     if mfcc:
#         mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#         result = np.hstack((result, mfccs))
#     if chroma:
#         stft = np.abs(librosa.stft(X))
#         chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, chroma_features))
#     if mel:
#         mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
#         result = np.hstack((result, mel_features))

#     return result

# # Load data and split into training and testing sets
# def load_data(test_size=0.2, augment=False):
#     x, y = [], []
#     for file in glob.glob("/Users/sonika.n/PROJECT/ser/api/speech-emotion-recognition-ravdess-data/**/*.wav", recursive=True):
#         file_name = os.path.basename(file)
#         emotion = emotions.get(file_name.split("-")[2])
#         if emotion not in observed_emotions:
#             continue
#         feature = extract_feature(file_name=file, mfcc=True, chroma=True, mel=True)
#         x.append(feature)
#         y.append(emotion)
#         if augment:
#             augmented_feature = add_noise(feature)
#             x.append(augmented_feature)
#             y.append(emotion)

#     if not x:
#         raise ValueError("No valid audio files found. Please check the dataset path and file format.")
    
#     return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# # Train model on dataset
# x_train, x_test, y_train, y_test = load_data(test_size=0.25, augment=True)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # Initialize and train the model
# model = MLPClassifier(alpha=0.001, batch_size=64, epsilon=1e-08, hidden_layer_sizes=(256, 128, 64), learning_rate='adaptive', max_iter=800)
# model.fit(x_train, y_train)

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     duration = 5
#     sample_rate = 22050
#     print("Recording...")

#     # Record live audio on button press
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()

#     # Extract features and scale
#     feature = extract_feature(audio=audio, sample_rate=sample_rate, mfcc=True, chroma=True, mel=True)
#     feature = scaler.transform(np.expand_dims(feature, axis=0))

#     # Predict emotion
#     prediction = model.predict(feature)
#     predicted_emotion = prediction[0]

#     # Define the path to save the audio file and analysis file
#     api_dir = os.path.join(os.path.dirname(__file__), "api")
#     os.makedirs(api_dir, exist_ok=True)
#     audio_file_name = os.path.join(api_dir, f"recording_{int(time.time())}.wav")
#     analysis_file_name = os.path.join(api_dir, f"analysis_{int(time.time())}.txt")

#     try:
#         # Save audio file
#         soundfile.write(audio_file_name, audio, sample_rate)
#         print(f"Audio file saved successfully at {audio_file_name}")

#         # Save analysis file
#         with open(analysis_file_name, "w") as f:
#             f.write(f"Predicted Emotion: {predicted_emotion}\n")
#             f.write(f"Audio File: {audio_file_name}\n")
#         print(f"Analysis file saved successfully at {analysis_file_name}")

#     except Exception as e:
#         return jsonify({'error': f'File generation failed: {str(e)}'})

#     return jsonify({'predicted_emotion': predicted_emotion, 'audio_file': audio_file_name, 'analysis_file': analysis_file_name})

# if __name__ == '__main__':
#     app.run(debug=True)





# import hashlib
# import json
# import time
# import wave
# import io
# import numpy as np
# from datetime import datetime
# import soundfile as sf
# import sounddevice as sd
# import base64
# from scipy.io import wavfile
# import logging
# from typing import Optional, Tuple, Any
# import os

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class AudioProcessingError(Exception):
#     """Custom exception for audio processing errors"""
#     pass

# class BlockchainError(Exception):
#     """Custom exception for blockchain-related errors"""
#     pass

# class Block:
#     def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
#         """
#         Initialize a new block in the blockchain
        
#         Args:
#             index: Block index
#             timestamp: Block creation timestamp
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
#             previous_hash: Hash of the previous block
#         """
#         self.index = index
#         self.timestamp = timestamp
#         self.audio_data = audio_data
#         self.emotion = emotion
#         self.previous_hash = previous_hash
#         self.nonce = 0
#         self.hash = self.calculate_hash()

#     def calculate_hash(self) -> str:
#         """Calculate the hash of the block using SHA-256"""
#         try:
#             block_string = json.dumps({
#                 "index": self.index,
#                 "timestamp": self.timestamp,
#                 "audio_data": str(self.audio_data),
#                 "emotion": self.emotion,
#                 "previous_hash": self.previous_hash,
#                 "nonce": self.nonce
#             }, sort_keys=True)
#             return hashlib.sha256(block_string.encode()).hexdigest()
#         except Exception as e:
#             logger.error(f"Hash calculation failed: {str(e)}")
#             raise BlockchainError("Failed to calculate block hash")

#     def mine_block(self, difficulty: int) -> None:
#         """
#         Mine the block with the given difficulty
        
#         Args:
#             difficulty: Number of leading zeros required in hash
#         """
#         try:
#             while self.hash[:difficulty] != '0' * difficulty:
#                 self.nonce += 1
#                 self.hash = self.calculate_hash()
#             logger.info(f"Block mined: {self.hash}")
#         except Exception as e:
#             logger.error(f"Block mining failed: {str(e)}")
#             raise BlockchainError("Failed to mine block")

# class EmotionAudioBlockchain:
#     def __init__(self, difficulty: int = 2):
#         """
#         Initialize the blockchain
        
#         Args:
#             difficulty: Mining difficulty (default=2)
#         """
#         self.chain: list = []
#         self.difficulty = difficulty
#         self.create_genesis_block()

#     def create_genesis_block(self) -> None:
#         """Create the first block in the chain"""
#         try:
#             genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
#             genesis_block.mine_block(self.difficulty)
#             self.chain.append(genesis_block)
#             logger.info("Genesis block created")
#         except Exception as e:
#             logger.error(f"Genesis block creation failed: {str(e)}")
#             raise BlockchainError("Failed to create genesis block")

#     def get_latest_block(self) -> Block:
#         """Get the most recent block in the chain"""
#         if not self.chain:
#             raise BlockchainError("Blockchain is empty")
#         return self.chain[-1]

#     def add_block(self, audio_data: str, emotion: str) -> Block:
#         """
#         Add a new block to the chain
        
#         Args:
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
        
#         Returns:
#             Block: The newly created and mined block
#         """
#         try:
#             new_block = Block(
#                 len(self.chain),
#                 str(datetime.now()),
#                 audio_data,
#                 emotion,
#                 self.get_latest_block().hash
#             )
#             new_block.mine_block(self.difficulty)
#             self.chain.append(new_block)
#             logger.info(f"New block added at index {new_block.index}")
#             return new_block
#         except Exception as e:
#             logger.error(f"Failed to add block: {str(e)}")
#             raise BlockchainError("Failed to add new block")

#     def is_chain_valid(self) -> bool:
#         """Verify the integrity of the blockchain"""
#         try:
#             for i in range(1, len(self.chain)):
#                 current_block = self.chain[i]
#                 previous_block = self.chain[i-1]

#                 if current_block.hash != current_block.calculate_hash():
#                     logger.error(f"Invalid hash at block {i}")
#                     return False

#                 if current_block.previous_hash != previous_block.hash:
#                     logger.error(f"Chain broken at block {i}")
#                     return False

#             return True
#         except Exception as e:
#             logger.error(f"Chain validation failed: {str(e)}")
#             return False

#     def save_to_file(self, filename: str = "blockchain_data.json") -> None:
#         """Save the blockchain to a JSON file"""
#         try:
#             blockchain_data = []
#             for block in self.chain:
#                 block_data = {
#                     'index': block.index,
#                     'timestamp': block.timestamp,
#                     'audio_data': block.audio_data,
#                     'emotion': block.emotion,
#                     'previous_hash': block.previous_hash,
#                     'hash': block.hash,
#                     'nonce': block.nonce
#                 }
#                 blockchain_data.append(block_data)

#             with open(filename, 'w') as f:
#                 json.dump(blockchain_data, f, indent=4)
#             logger.info(f"Blockchain saved to {filename}")
#         except Exception as e:
#             logger.error(f"Failed to save blockchain: {str(e)}")
#             raise BlockchainError("Failed to save blockchain to file")

#     def load_from_file(self, filename: str = "blockchain_data.json") -> None:
#         """Load the blockchain from a JSON file"""
#         try:
#             if not os.path.exists(filename):
#                 logger.info(f"No blockchain file found at {filename}")
#                 return

#             with open(filename, 'r') as f:
#                 blockchain_data = json.load(f)

#             self.chain = []
#             for block_data in blockchain_data:
#                 block = Block(
#                     block_data['index'],
#                     block_data['timestamp'],
#                     block_data['audio_data'],
#                     block_data['emotion'],
#                     block_data['previous_hash']
#                 )
#                 block.hash = block_data['hash']
#                 block.nonce = block_data['nonce']
#                 self.chain.append(block)
#             logger.info(f"Blockchain loaded from {filename}")
#         except Exception as e:
#             logger.error(f"Failed to load blockchain: {str(e)}")
#             raise BlockchainError("Failed to load blockchain from file")

# def encode_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
#     """Encode audio data to base64 string"""
#     try:
#         buffer = io.BytesIO()
#         sf.write(buffer, audio_data, sample_rate, format='WAV')
#         audio_bytes = buffer.getvalue()
#         return base64.b64encode(audio_bytes).decode('utf-8')
#     except Exception as e:
#         logger.error(f"Audio encoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to encode audio data")

# def decode_audio(audio_base64: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#     """Decode base64 string back to audio data"""
#     try:
#         audio_bytes = base64.b64decode(audio_base64)
#         buffer = io.BytesIO(audio_bytes)
#         audio_data, sample_rate = sf.read(buffer)
#         return audio_data, sample_rate
#     except Exception as e:
#         logger.error(f"Audio decoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to decode audio data")

# class EmotionRecognizer:
#     def __init__(self, model_path: str):
#         """
#         Initialize the emotion recognizer
        
#         Args:
#             model_path: Path to the trained model file
#         """
#         try:
#             # Here you would load your trained model
#             # self.model = load_model(model_path)
#             # self.label_encoder = ... # Load your label encoder
#             pass
#         except Exception as e:
#             logger.error(f"Failed to initialize emotion recognizer: {str(e)}")
#             raise Exception("Failed to initialize emotion recognizer")

#     def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
#         """
#         Predict emotion from audio data
        
#         Args:
#             audio_data: Audio signal data
#             sample_rate: Audio sample rate
            
#         Returns:
#             str: Predicted emotion
#         """
#         try:
#             # Add your emotion recognition logic here
#             # This is a placeholder that returns a random emotion
#             emotions = ["happy", "sad", "angry", "neutral"]
#             return np.random.choice(emotions)
#         except Exception as e:
#             logger.error(f"Emotion prediction failed: {str(e)}")
#             raise Exception("Failed to predict emotion")

# def live_recognition_with_blockchain(blockchain: EmotionAudioBlockchain, 
#                                    emotion_recognizer: EmotionRecognizer,
#                                    duration: int = 5, 
#                                    sample_rate: int = 22050) -> None:
#     """Record audio, perform emotion recognition, and store in blockchain"""
#     try:
#         logger.info("Starting audio recording...")
#         print(f"Please speak for {duration} seconds...")
        
#         # Record audio
#         audio = sd.rec(int(duration * sample_rate), 
#                       samplerate=sample_rate, 
#                       channels=1, 
#                       dtype='float32')
#         sd.wait()
#         audio = audio.flatten()

#         logger.info("Recording finished. Processing...")

#         # Predict emotion
#         predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)

#         # Encode audio data
#         audio_base64 = encode_audio(audio, sample_rate)

#         # Add to blockchain
#         block = blockchain.add_block(audio_base64, predicted_emotion)
        
#         print(f"Predicted Emotion: {predicted_emotion}")
#         print(f"Block Hash: {block.hash}")
#         print(f"Block Index: {block.index}")
        
#         # Save blockchain to file
#         blockchain.save_to_file()
        
#     except Exception as e:
#         logger.error(f"Live recognition failed: {str(e)}")
#         print(f"An error occurred: {str(e)}")

# def play_audio_from_block(block: Block) -> None:
#     """Play audio from a blockchain block"""
#     try:
#         if block.audio_data == "Genesis Block":
#             logger.info("Cannot play audio from genesis block")
#             print("Cannot play audio from genesis block")
#             return
            
#         # Decode and play the audio
#         audio_data, sample_rate = decode_audio(block.audio_data)
#         sd.play(audio_data, sample_rate)
#         sd.wait()
        
#     except Exception as e:
#         logger.error(f"Failed to play audio: {str(e)}")
#         print(f"An error occurred while playing audio: {str(e)}")

# def main():
#     try:
#         # Initialize blockchain and emotion recognizer
#         blockchain = EmotionAudioBlockchain(difficulty=2)
#         emotion_recognizer = EmotionRecognizer(model_path="path_to_your_model.h5")
        
#         # Load existing blockchain if available
#         blockchain.load_from_file()
        
#         while True:
#             print("\n=== Emotion Recognition Blockchain System ===")
#             print("1. Record new audio")
#             print("2. View all recordings")
#             print("3. Verify blockchain")
#             print("4. Exit")
            
#             try:
#                 choice = input("Enter your choice (1-4): ").strip()
                
#                 if choice == '1':
#                     live_recognition_with_blockchain(blockchain, emotion_recognizer)
                
#                 elif choice == '2':
#                     print("\nStored Recordings:")
#                     for block in blockchain.chain[1:]:  # Skip genesis block
#                         print(f"\nBlock {block.index}")
#                         print(f"Timestamp: {block.timestamp}")
#                         print(f"Emotion: {block.emotion}")
#                         print(f"Hash: {block.hash}")
                        
#                         if block.index > 0:
#                             play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
#                             if play_choice == 'y':
#                                 print("Playing audio...")
#                                 play_audio_from_block(block)
                
#                 elif choice == '3':
#                     if blockchain.is_chain_valid():
#                         print("\nBlockchain is valid!")
#                     else:
#                         print("\nBlockchain validation failed!")
                
#                 elif choice == '4':
#                     print("\nExiting...")
#                     break
                
#                 else:
#                     print("\nInvalid choice. Please try again.")
            
#             except KeyboardInterrupt:
#                 print("\nOperation cancelled by user")
#             except Exception as e:
#                 logger.error(f"Error in main loop: {str(e)}")
#                 print(f"An error occurred: {str(e)}")
                
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         print(f"Failed to start application: {str(e)}")

# if __name__ == "__main__":
#     main()


# import hashlib
# import json
# import time
# import wave
# import io
# import numpy as np
# from datetime import datetime
# import soundfile as sf
# import sounddevice as sd
# import base64
# from scipy.io import wavfile
# import logging
# from typing import Optional, Tuple, Any
# import os

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class AudioProcessingError(Exception):
#     """Custom exception for audio processing errors"""
#     pass

# class BlockchainError(Exception):
#     """Custom exception for blockchain-related errors"""
#     pass

# class Block:
#     def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
#         """
#         Initialize a new block in the blockchain
        
#         Args:
#             index: Block index
#             timestamp: Block creation timestamp
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
#             previous_hash: Hash of the previous block
#         """
#         self.index = index
#         self.timestamp = timestamp
#         self.audio_data = audio_data
#         self.emotion = emotion
#         self.previous_hash = previous_hash
#         self.nonce = 0
#         self.hash = self.calculate_hash()

#     def calculate_hash(self) -> str:
#         """Calculate the hash of the block using SHA-256"""
#         try:
#             block_string = json.dumps({
#                 "index": self.index,
#                 "timestamp": self.timestamp,
#                 "audio_data": str(self.audio_data),
#                 "emotion": self.emotion,
#                 "previous_hash": self.previous_hash,
#                 "nonce": self.nonce
#             }, sort_keys=True)
#             return hashlib.sha256(block_string.encode()).hexdigest()
#         except Exception as e:
#             logger.error(f"Hash calculation failed: {str(e)}")
#             raise BlockchainError("Failed to calculate block hash")

#     def mine_block(self, difficulty: int) -> None:
#         """
#         Mine the block with the given difficulty
        
#         Args:
#             difficulty: Number of leading zeros required in hash
#         """
#         try:
#             while self.hash[:difficulty] != '0' * difficulty:
#                 self.nonce += 1
#                 self.hash = self.calculate_hash()
#             logger.info(f"Block mined: {self.hash}")
#         except Exception as e:
#             logger.error(f"Block mining failed: {str(e)}")
#             raise BlockchainError("Failed to mine block")

# class EmotionAudioBlockchain:
#     def __init__(self, difficulty: int = 2):
#         """
#         Initialize the blockchain
        
#         Args:
#             difficulty: Mining difficulty (default=2)
#         """
#         self.chain: list = []
#         self.difficulty = difficulty
#         self.create_genesis_block()

#     def create_genesis_block(self) -> None:
#         """Create the first block in the chain"""
#         try:
#             genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
#             genesis_block.mine_block(self.difficulty)
#             self.chain.append(genesis_block)
#             logger.info("Genesis block created")
#         except Exception as e:
#             logger.error(f"Genesis block creation failed: {str(e)}")
#             raise BlockchainError("Failed to create genesis block")

#     def get_latest_block(self) -> Block:
#         """Get the most recent block in the chain"""
#         if not self.chain:
#             raise BlockchainError("Blockchain is empty")
#         return self.chain[-1]

#     def add_block(self, audio_data: str, emotion: str) -> Block:
#         """
#         Add a new block to the chain
        
#         Args:
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
        
#         Returns:
#             Block: The newly created and mined block
#         """
#         try:
#             new_block = Block(
#                 len(self.chain),
#                 str(datetime.now()),
#                 audio_data,
#                 emotion,
#                 self.get_latest_block().hash
#             )
#             new_block.mine_block(self.difficulty)
#             self.chain.append(new_block)
#             logger.info(f"New block added at index {new_block.index}")
#             return new_block
#         except Exception as e:
#             logger.error(f"Failed to add block: {str(e)}")
#             raise BlockchainError("Failed to add new block")

#     def is_chain_valid(self) -> bool:
#         """Verify the integrity of the blockchain"""
#         try:
#             for i in range(1, len(self.chain)):
#                 current_block = self.chain[i]
#                 previous_block = self.chain[i-1]

#                 if current_block.hash != current_block.calculate_hash():
#                     logger.error(f"Invalid hash at block {i}")
#                     return False

#                 if current_block.previous_hash != previous_block.hash:
#                     logger.error(f"Chain broken at block {i}")
#                     return False

#             return True
#         except Exception as e:
#             logger.error(f"Chain validation failed: {str(e)}")
#             return False

#     def save_to_file(self, filename: str = "blockchain_data.json") -> None:
#         """Save the blockchain to a JSON file"""
#         try:
#             blockchain_data = []
#             for block in self.chain:
#                 block_data = {
#                     'index': block.index,
#                     'timestamp': block.timestamp,
#                     'audio_data': block.audio_data,
#                     'emotion': block.emotion,
#                     'previous_hash': block.previous_hash,
#                     'hash': block.hash,
#                     'nonce': block.nonce
#                 }
#                 blockchain_data.append(block_data)

#             with open(filename, 'w') as f:
#                 json.dump(blockchain_data, f, indent=4)
#             logger.info(f"Blockchain saved to {filename}")
#         except Exception as e:
#             logger.error(f"Failed to save blockchain: {str(e)}")
#             raise BlockchainError("Failed to save blockchain to file")

#     def load_from_file(self, filename: str = "blockchain_data.json") -> None:
#         """Load the blockchain from a JSON file"""
#         try:
#             if not os.path.exists(filename):
#                 logger.info(f"No blockchain file found at {filename}")
#                 return

#             with open(filename, 'r') as f:
#                 blockchain_data = json.load(f)

#             self.chain = []
#             for block_data in blockchain_data:
#                 block = Block(
#                     block_data['index'],
#                     block_data['timestamp'],
#                     block_data['audio_data'],
#                     block_data['emotion'],
#                     block_data['previous_hash']
#                 )
#                 block.hash = block_data['hash']
#                 block.nonce = block_data['nonce']
#                 self.chain.append(block)
#             logger.info(f"Blockchain loaded from {filename}")
#         except Exception as e:
#             logger.error(f"Failed to load blockchain: {str(e)}")
#             raise BlockchainError("Failed to load blockchain from file")

# def encode_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
#     """Encode audio data to base64 string"""
#     try:
#         buffer = io.BytesIO()
#         sf.write(buffer, audio_data, sample_rate, format='WAV')
#         audio_bytes = buffer.getvalue()
#         return base64.b64encode(audio_bytes).decode('utf-8')
#     except Exception as e:
#         logger.error(f"Audio encoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to encode audio data")

# def decode_audio(audio_base64: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#     """Decode base64 string back to audio data"""
#     try:
#         audio_bytes = base64.b64decode(audio_base64)
#         buffer = io.BytesIO(audio_bytes)
#         audio_data, sample_rate = sf.read(buffer)
#         return audio_data, sample_rate
#     except Exception as e:
#         logger.error(f"Audio decoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to decode audio data")

# class EmotionRecognizer:
#     def __init__(self, model_path: str):
#         """
#         Initialize the emotion recognizer
        
#         Args:
#             model_path: Path to the trained model file
#         """
#         try:
#             # Here you would load your trained model
#             # self.model = load_model(model_path)
#             # self.label_encoder = ... # Load your label encoder
#             pass
#         except Exception as e:
#             logger.error(f"Failed to initialize emotion recognizer: {str(e)}")
#             raise Exception("Failed to initialize emotion recognizer")

#     def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
#         """
#         Predict emotion from audio data
        
#         Args:
#             audio_data: Audio signal data
#             sample_rate: Audio sample rate
            
#         Returns:
#             str: Predicted emotion
#         """
#         try:
#             # Add your emotion recognition logic here
#             # This is a placeholder that returns a random emotion
#             emotions = ["happy", "sad", "angry", "neutral"]
#             return np.random.choice(emotions)
#         except Exception as e:
#             logger.error(f"Emotion prediction failed: {str(e)}")
#             raise Exception("Failed to predict emotion")

# def live_recognition_with_blockchain(blockchain: EmotionAudioBlockchain, 
#                                    emotion_recognizer: EmotionRecognizer,
#                                    duration: int = 5, 
#                                    sample_rate: int = 22050) -> None:
#     """Record audio, perform emotion recognition, and store in blockchain"""
#     try:
#         logger.info("Starting audio recording...")
#         print(f"Please speak for {duration} seconds...")
        
#         # Record audio
#         audio = sd.rec(int(duration * sample_rate), 
#                       samplerate=sample_rate, 
#                       channels=1, 
#                       dtype='float32')
#         sd.wait()
#         audio = audio.flatten()

#         logger.info("Recording finished. Processing...")

#         # Predict emotion
#         predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)

#         # Encode audio data
#         audio_base64 = encode_audio(audio, sample_rate)

#         # Add to blockchain
#         block = blockchain.add_block(audio_base64, predicted_emotion)
        
#         print(f"Predicted Emotion: {predicted_emotion}")
#         print(f"Block Hash: {block.hash}")
#         print(f"Block Index: {block.index}")
        
#         # Save blockchain to file
#         blockchain.save_to_file()
        
#     except Exception as e:
#         logger.error(f"Live recognition failed: {str(e)}")
#         print(f"An error occurred: {str(e)}")

# def play_audio_from_block(block: Block) -> None:
#     """Play audio from a blockchain block"""
#     try:
#         if block.audio_data == "Genesis Block":
#             logger.info("Cannot play audio from genesis block")
#             print("Cannot play audio from genesis block")
#             return
            
#         # Decode and play the audio
#         audio_data, sample_rate = decode_audio(block.audio_data)
#         sd.play(audio_data, sample_rate)
#         sd.wait()
        
#     except Exception as e:
#         logger.error(f"Failed to play audio: {str(e)}")
#         print(f"An error occurred while playing audio: {str(e)}")

# def main():
#     try:
#         # Initialize blockchain and emotion recognizer
#         blockchain = EmotionAudioBlockchain(difficulty=2)
#         emotion_recognizer = EmotionRecognizer(model_path="path_to_your_model.h5")
        
#         # Load existing blockchain if available
#         blockchain.load_from_file()
        
#         while True:
#             print("\n=== Emotion Recognition Blockchain System ===")
#             print("1. Record new audio")
#             print("2. View all recordings")
#             print("3. Verify blockchain")
#             print("4. Exit")
            
#             try:
#                 choice = input("Enter your choice (1-4): ").strip()
                
#                 if choice == '1':
#                     live_recognition_with_blockchain(blockchain, emotion_recognizer)
                
#                 elif choice == '2':
#                     print("\nStored Recordings:")
#                     for block in blockchain.chain[1:]:  # Skip genesis block
#                         print(f"\nBlock {block.index}")
#                         print(f"Timestamp: {block.timestamp}")
#                         print(f"Emotion: {block.emotion}")
#                         print(f"Hash: {block.hash}")
                        
#                         if block.index > 0:
#                             play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
#                             if play_choice == 'y':
#                                 print("Playing audio...")
#                                 play_audio_from_block(block)
                
#                 elif choice == '3':
#                     if blockchain.is_chain_valid():
#                         print("\nBlockchain is valid!")
#                     else:
#                         print("\nBlockchain validation failed!")
                
#                 elif choice == '4':
#                     print("\nExiting...")
#                     break
                
#                 else:
#                     print("\nInvalid choice. Please try again.")
            
#             except KeyboardInterrupt:
#                 print("\nOperation cancelled by user")
#             except Exception as e:
#                 logger.error(f"Error in main loop: {str(e)}")
#                 print(f"An error occurred: {str(e)}")
                
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         print(f"Failed to start application: {str(e)}")

# if __name__ == "__main__":
#     main()


# import hashlib
# import json
# import time
# import wave
# import io
# import numpy as np
# from datetime import datetime
# import soundfile as sf
# import sounddevice as sd
# import base64
# from scipy.io import wavfile
# import logging
# from typing import Optional, Tuple, Any
# import os
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import librosa
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import glob

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Flask setup
# app = Flask(__name__)
# CORS(app)

# # Emotions dictionary for labeling data
# emotions = {
#     '01': 'neutral',
#     '02': 'calm',
#     '03': 'happy',
#     '04': 'sad',
#     '05': 'angry',
#     '06': 'fearful',
#     '07': 'disgust',
#     '08': 'surprised'
# }
# observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# class AudioProcessingError(Exception):
#     pass

# class BlockchainError(Exception):
#     pass

# class Block:
#     def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
#         self.index = index
#         self.timestamp = timestamp
#         self.audio_data = audio_data
#         self.emotion = emotion
#         self.previous_hash = previous_hash
#         self.nonce = 0
#         self.hash = self.calculate_hash()

#     def calculate_hash(self) -> str:
#         block_string = json.dumps({
#             "index": self.index,
#             "timestamp": self.timestamp,
#             "audio_data": str(self.audio_data),
#             "emotion": self.emotion,
#             "previous_hash": self.previous_hash,
#             "nonce": self.nonce
#         }, sort_keys=True)
#         return hashlib.sha256(block_string.encode()).hexdigest()

#     def mine_block(self, difficulty: int) -> None:
#         while self.hash[:difficulty] != '0' * difficulty:
#             self.nonce += 1
#             self.hash = self.calculate_hash()

# class EmotionAudioBlockchain:
#     def __init__(self, difficulty: int = 2):
#         self.chain: list = []
#         self.difficulty = difficulty
#         self.create_genesis_block()

#     def create_genesis_block(self) -> None:
#         genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
#         genesis_block.mine_block(self.difficulty)
#         self.chain.append(genesis_block)

#     def get_latest_block(self) -> Block:
#         return self.chain[-1]

#     def add_block(self, audio_data: str, emotion: str) -> Block:
#         new_block = Block(
#             len(self.chain),
#             str(datetime.now()),
#             audio_data,
#             emotion,
#             self.get_latest_block().hash
#         )
#         new_block.mine_block(self.difficulty)
#         self.chain.append(new_block)
#         return new_block

#     def is_chain_valid(self) -> bool:
#         for i in range(1, len(self.chain)):
#             current_block = self.chain[i]
#             previous_block = self.chain[i - 1]
#             if current_block.hash != current_block.calculate_hash():
#                 return False
#             if current_block.previous_hash != previous_block.hash:
#                 return False
#         return True

# # Encode and decode functions for audio data
# def encode_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
#     buffer = io.BytesIO()
#     sf.write(buffer, audio_data, sample_rate, format='WAV')
#     audio_bytes = buffer.getvalue()
#     return base64.b64encode(audio_bytes).decode('utf-8')

# def decode_audio(audio_base64: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#     audio_bytes = base64.b64decode(audio_base64)
#     buffer = io.BytesIO(audio_bytes)
#     audio_data, sample_rate = sf.read(buffer)
#     return audio_data, sample_rate

# # EmotionRecognizer class with dummy model
# class EmotionRecognizer:
#     def __init__(self, model=None):
#         self.model = model or MLPClassifier(alpha=0.001, batch_size=64, epsilon=1e-08,
#                                             hidden_layer_sizes=(256, 128, 64), learning_rate='adaptive', max_iter=500)
#         self.scaler = StandardScaler()

#     def extract_features(self, audio, sample_rate=22050):
#         mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
#         return mfccs

#     def train_model(self, x_train, y_train):
#         x_train = self.scaler.fit_transform(x_train)
#         self.model.fit(x_train, y_train)

#     def predict_emotion(self, audio, sample_rate=22050):
#         features = self.extract_features(audio, sample_rate)
#         features = self.scaler.transform([features])
#         return self.model.predict(features)[0]

# def live_recognition_with_blockchain(blockchain, emotion_recognizer, duration=5, sample_rate=22050):
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()
#     predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
#     audio_base64 = encode_audio(audio, sample_rate)
#     block = blockchain.add_block(audio_base64, predicted_emotion)
#     return {"emotion": predicted_emotion, "hash": block.hash, "index": block.index}

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     duration = request.json.get("duration", 5)
#     sample_rate = 22050
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
#     sd.wait()
#     audio = audio.flatten()
#     emotion_recognizer = EmotionRecognizer()
#     predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
#     audio_file_name = f"recording_{int(time.time())}.wav"
#     sf.write(audio_file_name, audio, sample_rate)
#     return jsonify({'predicted_emotion': predicted_emotion, 'audio_file': audio_file_name})

# # Initialize blockchain and model, and run the server
# if __name__ == '__main__':
#     blockchain = EmotionAudioBlockchain(difficulty=2)
#     emotion_recognizer = EmotionRecognizer()

#     # Load data, preprocess, and train model (example)
#     x, y = [], []
#     for file in glob.glob("/Users/sonika.n/Desktop/MINI PROJECT/api/speech-emotion-recognition-ravdess-data/**/*.wav", recursive=True):
#         emotion = emotions.get(file.split("-")[2])
#         if emotion not in observed_emotions:
#             continue
#         audio, sr = librosa.load(file, sr=22050)
#         features = emotion_recognizer.extract_features(audio, sr)
#         x.append(features)
#         y.append(emotion)
#     x_train, x_test, y_train, y_test = train_test_split(np.array(x), y, test_size=0.25, random_state=42)
#     emotion_recognizer.train_model(x_train, y_train)

#     # Start the Flask app
#     app.run(debug=True)



# import hashlib
# import json
# import time
# import wave
# import io
# import numpy as np
# from datetime import datetime
# import soundfile as sf
# import sounddevice as sd
# import base64
# from scipy.io import wavfile
# import logging
# from typing import Optional, Tuple, Any
# import os

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class AudioProcessingError(Exception):
#     """Custom exception for audio processing errors"""
#     pass

# class BlockchainError(Exception):
#     """Custom exception for blockchain-related errors"""
#     pass

# class Block:
#     def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
#         """
#         Initialize a new block in the blockchain
        
#         Args:
#             index: Block index
#             timestamp: Block creation timestamp
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
#             previous_hash: Hash of the previous block
#         """
#         self.index = index
#         self.timestamp = timestamp
#         self.audio_data = audio_data
#         self.emotion = emotion
#         self.previous_hash = previous_hash
#         self.nonce = 0
#         self.hash = self.calculate_hash()

#     def calculate_hash(self) -> str:
#         """Calculate the hash of the block using SHA-256"""
#         try:
#             block_string = json.dumps({
#                 "index": self.index,
#                 "timestamp": self.timestamp,
#                 "audio_data": str(self.audio_data),
#                 "emotion": self.emotion,
#                 "previous_hash": self.previous_hash,
#                 "nonce": self.nonce
#             }, sort_keys=True)
#             return hashlib.sha256(block_string.encode()).hexdigest()
#         except Exception as e:
#             logger.error(f"Hash calculation failed: {str(e)}")
#             raise BlockchainError("Failed to calculate block hash")

#     def mine_block(self, difficulty: int) -> None:
#         """
#         Mine the block with the given difficulty
        
#         Args:
#             difficulty: Number of leading zeros required in hash
#         """
#         try:
#             while self.hash[:difficulty] != '0' * difficulty:
#                 self.nonce += 1
#                 self.hash = self.calculate_hash()
#             logger.info(f"Block mined: {self.hash}")
#         except Exception as e:
#             logger.error(f"Block mining failed: {str(e)}")
#             raise BlockchainError("Failed to mine block")

# class EmotionAudioBlockchain:
#     def __init__(self, difficulty: int = 2):
#         """
#         Initialize the blockchain
        
#         Args:
#             difficulty: Mining difficulty (default=2)
#         """
#         self.chain: list = []
#         self.difficulty = difficulty
#         self.create_genesis_block()

#     def create_genesis_block(self) -> None:
#         """Create the first block in the chain"""
#         try:
#             genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
#             genesis_block.mine_block(self.difficulty)
#             self.chain.append(genesis_block)
#             logger.info("Genesis block created")
#         except Exception as e:
#             logger.error(f"Genesis block creation failed: {str(e)}")
#             raise BlockchainError("Failed to create genesis block")

#     def get_latest_block(self) -> Block:
#         """Get the most recent block in the chain"""
#         if not self.chain:
#             raise BlockchainError("Blockchain is empty")
#         return self.chain[-1]

#     def add_block(self, audio_data: str, emotion: str) -> Block:
#         """
#         Add a new block to the chain
        
#         Args:
#             audio_data: Base64 encoded audio data
#             emotion: Detected emotion
        
#         Returns:
#             Block: The newly created and mined block
#         """
#         try:
#             new_block = Block(
#                 len(self.chain),
#                 str(datetime.now()),
#                 audio_data,
#                 emotion,
#                 self.get_latest_block().hash
#             )
#             new_block.mine_block(self.difficulty)
#             self.chain.append(new_block)
#             logger.info(f"New block added at index {new_block.index}")
#             return new_block
#         except Exception as e:
#             logger.error(f"Failed to add block: {str(e)}")
#             raise BlockchainError("Failed to add new block")

#     def is_chain_valid(self) -> bool:
#         """Verify the integrity of the blockchain"""
#         try:
#             for i in range(1, len(self.chain)):
#                 current_block = self.chain[i]
#                 previous_block = self.chain[i-1]

#                 if current_block.hash != current_block.calculate_hash():
#                     logger.error(f"Invalid hash at block {i}")
#                     return False

#                 if current_block.previous_hash != previous_block.hash:
#                     logger.error(f"Chain broken at block {i}")
#                     return False

#             return True
#         except Exception as e:
#             logger.error(f"Chain validation failed: {str(e)}")
#             return False

#     def save_to_file(self, filename: str = "blockchain_data.json") -> None:
#         """Save the blockchain to a JSON file"""
#         try:
#             blockchain_data = []
#             for block in self.chain:
#                 block_data = {
#                     'index': block.index,
#                     'timestamp': block.timestamp,
#                     'audio_data': block.audio_data,
#                     'emotion': block.emotion,
#                     'previous_hash': block.previous_hash,
#                     'hash': block.hash,
#                     'nonce': block.nonce
#                 }
#                 blockchain_data.append(block_data)

#             with open(filename, 'w') as f:
#                 json.dump(blockchain_data, f, indent=4)
#             logger.info(f"Blockchain saved to {filename}")
#         except Exception as e:
#             logger.error(f"Failed to save blockchain: {str(e)}")
#             raise BlockchainError("Failed to save blockchain to file")

#     def load_from_file(self, filename: str = "blockchain_data.json") -> None:
#         """Load the blockchain from a JSON file"""
#         try:
#             if not os.path.exists(filename):
#                 logger.info(f"No blockchain file found at {filename}")
#                 return

#             with open(filename, 'r') as f:
#                 blockchain_data = json.load(f)

#             self.chain = []
#             for block_data in blockchain_data:
#                 block = Block(
#                     block_data['index'],
#                     block_data['timestamp'],
#                     block_data['audio_data'],
#                     block_data['emotion'],
#                     block_data['previous_hash']
#                 )
#                 block.hash = block_data['hash']
#                 block.nonce = block_data['nonce']
#                 self.chain.append(block)
#             logger.info(f"Blockchain loaded from {filename}")
#         except Exception as e:
#             logger.error(f"Failed to load blockchain: {str(e)}")
#             raise BlockchainError("Failed to load blockchain from file")

# def encode_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
#     """Encode audio data to base64 string"""
#     try:
#         buffer = io.BytesIO()
#         sf.write(buffer, audio_data, sample_rate, format='WAV')
#         audio_bytes = buffer.getvalue()
#         return base64.b64encode(audio_bytes).decode('utf-8')
#     except Exception as e:
#         logger.error(f"Audio encoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to encode audio data")

# def decode_audio(audio_base64: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#     """Decode base64 string back to audio data"""
#     try:
#         audio_bytes = base64.b64decode(audio_base64)
#         buffer = io.BytesIO(audio_bytes)
#         audio_data, sample_rate = sf.read(buffer)
#         return audio_data, sample_rate
#     except Exception as e:
#         logger.error(f"Audio decoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to decode audio data")

# class EmotionRecognizer:
#     def __init__(self, model_path: str):
#         """
#         Initialize the emotion recognizer
        
#         Args:
#             model_path: Path to the trained model file
#         """
#         try:
#             # Here you would load your trained model
#             # self.model = load_model(model_path)
#             # self.label_encoder = ... # Load your label encoder
#             pass
#         except Exception as e:
#             logger.error(f"Failed to initialize emotion recognizer: {str(e)}")
#             raise Exception("Failed to initialize emotion recognizer")

#     def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
#         """
#         Predict emotion from audio data
        
#         Args:
#             audio_data: Audio signal data
#             sample_rate: Audio sample rate
            
#         Returns:
#             str: Predicted emotion
#         """
#         try:
#             # Add your emotion recognition logic here
#             # This is a placeholder that returns a random emotion
#             emotions = ["happy", "sad", "angry", "neutral"]
#             return np.random.choice(emotions)
#         except Exception as e:
#             logger.error(f"Emotion prediction failed: {str(e)}")
#             raise Exception("Failed to predict emotion")

# def live_recognition_with_blockchain(blockchain: EmotionAudioBlockchain, 
#                                    emotion_recognizer: EmotionRecognizer,
#                                    duration: int = 5, 
#                                    sample_rate: int = 22050) -> None:
#     """Record audio, perform emotion recognition, and store in blockchain"""
#     try:
#         logger.info("Starting audio recording...")
#         print(f"Please speak for {duration} seconds...")
        
#         # Record audio
#         audio = sd.rec(int(duration * sample_rate), 
#                       samplerate=sample_rate, 
#                       channels=1, 
#                       dtype='float32')
#         sd.wait()
#         audio = audio.flatten()

#         logger.info("Recording finished. Processing...")

#         # Predict emotion
#         predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)

#         # Encode audio data
#         audio_base64 = encode_audio(audio, sample_rate)

#         # Add to blockchain
#         block = blockchain.add_block(audio_base64, predicted_emotion)
        
#         print(f"Predicted Emotion: {predicted_emotion}")
#         print(f"Block Hash: {block.hash}")
#         print(f"Block Index: {block.index}")
        
#         # Save blockchain to file
#         blockchain.save_to_file()
        
#     except Exception as e:
#         logger.error(f"Live recognition failed: {str(e)}")
#         print(f"An error occurred: {str(e)}")

# def play_audio_from_block(block: Block) -> None:
#     """Play audio from a blockchain block"""
#     try:
#         if block.audio_data == "Genesis Block":
#             logger.info("Cannot play audio from genesis block")
#             print("Cannot play audio from genesis block")
#             return
            
#         # Decode and play the audio
#         audio_data, sample_rate = decode_audio(block.audio_data)
#         sd.play(audio_data, sample_rate)
#         sd.wait()
        
#     except Exception as e:
#         logger.error(f"Failed to play audio: {str(e)}")
#         print(f"An error occurred while playing audio: {str(e)}")

# def main():
#     try:
#         # Initialize blockchain and emotion recognizer
#         blockchain = EmotionAudioBlockchain(difficulty=2)
#         emotion_recognizer = EmotionRecognizer(model_path="path_to_your_model.h5")
        
#         # Load existing blockchain if available
#         blockchain.load_from_file()
        
#         while True:
#             print("\n=== Emotion Recognition Blockchain System ===")
#             print("1. Record new audio")
#             print("2. View all recordings")
#             print("3. Verify blockchain")
#             print("4. Exit")
            
#             try:
#                 choice = input("Enter your choice (1-4): ").strip()
                
#                 if choice == '1':
#                     live_recognition_with_blockchain(blockchain, emotion_recognizer)
                
#                 elif choice == '2':
#                     print("\nStored Recordings:")
#                     for block in blockchain.chain[1:]:  # Skip genesis block
#                         print(f"\nBlock {block.index}")
#                         print(f"Timestamp: {block.timestamp}")
#                         print(f"Emotion: {block.emotion}")
#                         print(f"Hash: {block.hash}")
                        
#                         if block.index > 0:
#                             play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
#                             if play_choice == 'y':
#                                 print("Playing audio...")
#                                 play_audio_from_block(block)
                
#                 elif choice == '3':
#                     if blockchain.is_chain_valid():
#                         print("\nBlockchain is valid!")
#                     else:
#                         print("\nBlockchain validation failed!")
                
#                 elif choice == '4':
#                     print("\nExiting...")
#                     break
                
#                 else:
#                     print("\nInvalid choice. Please try again.")
            
#             except KeyboardInterrupt:
#                 print("\nOperation cancelled by user")
#             except Exception as e:
#                 logger.error(f"Error in main loop: {str(e)}")
#                 print(f"An error occurred: {str(e)}")
                
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         print(f"Failed to start application: {str(e)}")

# if __name__ == "__main__":
#     main()


# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import hashlib
# import json
# import time
# import wave
# import io
# import numpy as np
# from datetime import datetime
# import soundfile as sf
# import sounddevice as sd
# import base64
# import librosa
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from typing import Optional, Tuple, Any
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class AudioProcessingError(Exception):
#     """Custom exception for audio processing errors"""
#     pass

# class BlockchainError(Exception):
#     """Custom exception for blockchain-related errors"""
#     pass

# def extract_feature(audio, sample_rate=22050, mfcc=True, chroma=True, mel=True):
#     """Extract features from audio data"""
#     try:
#         result = np.array([])
        
#         if mfcc:
#             mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
#             result = np.hstack((result, mfccs))
            
#         if chroma:
#             stft = np.abs(librosa.stft(audio))
#             chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, chroma_features))
            
#         if mel:
#             mel_features = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
#             result = np.hstack((result, mel_features))
            
#         return result
#     except Exception as e:
#         logger.error(f"Feature extraction failed: {str(e)}")
#         raise AudioProcessingError("Failed to extract audio features")

# class EmotionRecognizer:
#     def __init__(self):
#         """Initialize the emotion recognizer with CNN model"""
#         try:
#             self.height = 15
#             self.width = 12
#             self.channels = 1
#             self.scaler = StandardScaler()
#             self.label_encoder = LabelEncoder()
            
#             # Initialize labels
#             observed_emotions = ['calm', 'happy', 'fearful', 'disgust']
#             self.label_encoder.fit(observed_emotions)
            
#             # Create and compile the model
#             self.model = self._create_model()
            
#         except Exception as e:
#             logger.error(f"Failed to initialize emotion recognizer: {str(e)}")
#             raise Exception("Failed to initialize emotion recognizer")

#     def _create_model(self):
#         """Create the CNN model architecture"""
#         model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(self.height, self.width, self.channels)),
#             tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#             tf.keras.layers.MaxPooling2D((2, 2)),
#             tf.keras.layers.Flatten(),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
#         ])
        
#         model.compile(optimizer='adam',
#                      loss='sparse_categorical_crossentropy',
#                      metrics=['accuracy'])
#         return model

#     def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
#         """Predict emotion from audio data"""
#         try:
#             # Extract features
#             features = extract_feature(audio_data, sample_rate)
            
#             # Scale features
#             features = self.scaler.fit_transform(features.reshape(1, -1))
            
#             # Reshape for CNN
#             features = features.reshape(-1, self.height, self.width, self.channels)
            
#             # Predict
#             prediction = np.argmax(self.model.predict(features), axis=1)
#             emotion = self.label_encoder.inverse_transform(prediction)[0]
            
#             return emotion
#         except Exception as e:
#             logger.error(f"Emotion prediction failed: {str(e)}")
#             raise Exception("Failed to predict emotion")

# class Block:
#     def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
#         """Initialize a new block in the blockchain"""
#         self.index = index
#         self.timestamp = timestamp
#         self.audio_data = audio_data
#         self.emotion = emotion
#         self.previous_hash = previous_hash
#         self.nonce = 0
#         self.hash = self.calculate_hash()

#     def calculate_hash(self) -> str:
#         """Calculate the hash of the block using SHA-256"""
#         try:
#             block_string = json.dumps({
#                 "index": self.index,
#                 "timestamp": self.timestamp,
#                 "audio_data": str(self.audio_data),
#                 "emotion": self.emotion,
#                 "previous_hash": self.previous_hash,
#                 "nonce": self.nonce
#             }, sort_keys=True)
#             return hashlib.sha256(block_string.encode()).hexdigest()
#         except Exception as e:
#             logger.error(f"Hash calculation failed: {str(e)}")
#             raise BlockchainError("Failed to calculate block hash")

#     def mine_block(self, difficulty: int) -> None:
#         """Mine the block with the given difficulty"""
#         try:
#             while self.hash[:difficulty] != '0' * difficulty:
#                 self.nonce += 1
#                 self.hash = self.calculate_hash()
#             logger.info(f"Block mined: {self.hash}")
#         except Exception as e:
#             logger.error(f"Block mining failed: {str(e)}")
#             raise BlockchainError("Failed to mine block")

# class EmotionAudioBlockchain:
#     def __init__(self, difficulty: int = 2):
#         """Initialize the blockchain"""
#         self.chain = []
#         self.difficulty = difficulty
#         self.create_genesis_block()

#     def create_genesis_block(self) -> None:
#         """Create the first block in the chain"""
#         try:
#             genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
#             genesis_block.mine_block(self.difficulty)
#             self.chain.append(genesis_block)
#             logger.info("Genesis block created")
#         except Exception as e:
#             logger.error(f"Genesis block creation failed: {str(e)}")
#             raise BlockchainError("Failed to create genesis block")

#     def get_latest_block(self) -> Block:
#         """Get the most recent block in the chain"""
#         if not self.chain:
#             raise BlockchainError("Blockchain is empty")
#         return self.chain[-1]

#     def add_block(self, audio_data: str, emotion: str) -> Block:
#         """Add a new block to the chain"""
#         try:
#             new_block = Block(
#                 len(self.chain),
#                 str(datetime.now()),
#                 audio_data,
#                 emotion,
#                 self.get_latest_block().hash
#             )
#             new_block.mine_block(self.difficulty)
#             self.chain.append(new_block)
#             logger.info(f"New block added at index {new_block.index}")
#             return new_block
#         except Exception as e:
#             logger.error(f"Failed to add block: {str(e)}")
#             raise BlockchainError("Failed to add new block")

#     def is_chain_valid(self) -> bool:
#         """Verify the integrity of the blockchain"""
#         try:
#             for i in range(1, len(self.chain)):
#                 current_block = self.chain[i]
#                 previous_block = self.chain[i-1]

#                 if current_block.hash != current_block.calculate_hash():
#                     logger.error(f"Invalid hash at block {i}")
#                     return False

#                 if current_block.previous_hash != previous_block.hash:
#                     logger.error(f"Chain broken at block {i}")
#                     return False

#             return True
#         except Exception as e:
#             logger.error(f"Chain validation failed: {str(e)}")
#             return False

#     def save_to_file(self, filename: str = "blockchain_data.json") -> None:
#         """Save the blockchain to a JSON file"""
#         try:
#             blockchain_data = []
#             for block in self.chain:
#                 block_data = {
#                     'index': block.index,
#                     'timestamp': block.timestamp,
#                     'audio_data': block.audio_data,
#                     'emotion': block.emotion,
#                     'previous_hash': block.previous_hash,
#                     'hash': block.hash,
#                     'nonce': block.nonce
#                 }
#                 blockchain_data.append(block_data)

#             with open(filename, 'w') as f:
#                 json.dump(blockchain_data, f, indent=4)
#             logger.info(f"Blockchain saved to {filename}")
#         except Exception as e:
#             logger.error(f"Failed to save blockchain: {str(e)}")
#             raise BlockchainError("Failed to save blockchain to file")

#     def load_from_file(self, filename: str = "blockchain_data.json") -> None:
#         """Load the blockchain from a JSON file"""
#         try:
#             if not os.path.exists(filename):
#                 logger.info(f"No blockchain file found at {filename}")
#                 return

#             with open(filename, 'r') as f:
#                 blockchain_data = json.load(f)

#             self.chain = []
#             for block_data in blockchain_data:
#                 block = Block(
#                     block_data['index'],
#                     block_data['timestamp'],
#                     block_data['audio_data'],
#                     block_data['emotion'],
#                     block_data['previous_hash']
#                 )
#                 block.hash = block_data['hash']
#                 block.nonce = block_data['nonce']
#                 self.chain.append(block)
#             logger.info(f"Blockchain loaded from {filename}")
#         except Exception as e:
#             logger.error(f"Failed to load blockchain: {str(e)}")
#             raise BlockchainError("Failed to load blockchain from file")

# def encode_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
#     """Encode audio data to base64 string"""
#     try:
#         buffer = io.BytesIO()
#         sf.write(buffer, audio_data, sample_rate, format='WAV')
#         audio_bytes = buffer.getvalue()
#         return base64.b64encode(audio_bytes).decode('utf-8')
#     except Exception as e:
#         logger.error(f"Audio encoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to encode audio data")

# def decode_audio(audio_base64: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
#     """Decode base64 string back to audio data"""
#     try:
#         audio_bytes = base64.b64decode(audio_base64)
#         buffer = io.BytesIO(audio_bytes)
#         audio_data, sample_rate = sf.read(buffer)
#         return audio_data, sample_rate
#     except Exception as e:
#         logger.error(f"Audio decoding failed: {str(e)}")
#         raise AudioProcessingError("Failed to decode audio data")

# def live_recognition_with_blockchain(blockchain: EmotionAudioBlockchain, 
#                                    emotion_recognizer: EmotionRecognizer,
#                                    duration: int = 5, 
#                                    sample_rate: int = 22050) -> None:
#     """Record audio, perform emotion recognition, and store in blockchain"""
#     try:
#         logger.info("Starting audio recording...")
#         print(f"Please speak for {duration} seconds...")
        
#         # Record audio
#         audio = sd.rec(int(duration * sample_rate), 
#                       samplerate=sample_rate, 
#                       channels=1, 
#                       dtype='float32')
#         sd.wait()
#         audio = audio.flatten()

#         logger.info("Recording finished. Processing...")

#         # Predict emotion using the actual model
#         predicted_emotion = emotion_recognizer.predict_emotion(audio, sample_rate)

#         # Encode audio data
#         audio_base64 = encode_audio(audio, sample_rate)

#         # Add to blockchain
#         block = blockchain.add_block(audio_base64, predicted_emotion)
        
#         print(f"Predicted Emotion: {predicted_emotion}")
#         print(f"Block Hash: {block.hash}")
#         print(f"Block Index: {block.index}")
        
#         # Save blockchain to file
#         blockchain.save_to_file()
        
#     except Exception as e:
#         logger.error(f"Live recognition failed: {str(e)}")
#         print(f"An error occurred: {str(e)}")

# def play_audio_from_block(block: Block) -> None:
#     """Play audio from a blockchain block"""
#     try:
#         if block.audio_data == "Genesis Block":
#             logger.info("Cannot play audio from genesis block")
#             print("Cannot play audio from genesis block")
#             return
            
#         # Decode and play the audio
#         audio_data, sample_rate = decode_audio(block.audio_data)
#         sd.play(audio_data, sample_rate)
#         sd.wait()
        
#     except Exception as e:
#         logger.error(f"Failed to play audio: {str(e)}")
#         print(f"An error occurred while playing audio: {str(e)}")
# def main():
#     try:
#         print("Initializing Emotion Recognition Blockchain System...")
        
#         # Initialize blockchain with moderate difficulty
#         blockchain = EmotionAudioBlockchain(difficulty=2)
        
#         # Initialize emotion recognizer
#         print("Loading emotion recognition model...")
#         emotion_recognizer = EmotionRecognizer()
        
#         # Load existing blockchain if available
#         print("Loading existing blockchain data...")
#         blockchain.load_from_file()
        
#         print("System initialized successfully!")
        
#         while True:
#             print("\n=== Emotion Recognition Blockchain System ===")
#             print("1. Record new audio and analyze emotion")
#             print("2. View and play stored recordings")
#             print("3. Verify blockchain integrity")
#             print("4. Save blockchain to file")
#             print("5. View system statistics")
#             print("6. Exit")
            
#             try:
#                 choice = input("\nEnter your choice (1-6): ").strip()
                
#                 if choice == '1':
#                     print("\n=== Recording New Audio ===")
#                     duration = 5  # Default recording duration
#                     try:
#                         custom_duration = input("Enter recording duration in seconds (press Enter for default 5s): ").strip()
#                         if custom_duration:
#                             duration = int(custom_duration)
#                     except ValueError:
#                         print("Invalid duration. Using default 5 seconds.")
                    
#                     live_recognition_with_blockchain(blockchain, emotion_recognizer, duration=duration)
                
#                 elif choice == '2':
#                     print("\n=== Stored Recordings ===")
#                     if len(blockchain.chain) <= 1:
#                         print("No recordings found (only genesis block exists)")
#                         continue
                        
#                     for block in blockchain.chain[1:]:  # Skip genesis block
#                         print(f"\nBlock {block.index}")
#                         print(f"Timestamp: {block.timestamp}")
#                         print(f"Emotion: {block.emotion}")
#                         print(f"Hash: {block.hash[:15]}...")  # Show truncated hash
                        
#                         play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
#                         if play_choice == 'y':
#                             print("Playing audio...")
#                             play_audio_from_block(block)
#                             input("Press Enter to continue...")  # Pause after playback
                
#                 elif choice == '3':
#                     print("\n=== Verifying Blockchain ===")
#                     if blockchain.is_chain_valid():
#                         print("✓ Blockchain verification successful!")
#                         print(f"Total blocks: {len(blockchain.chain)}")
#                         print(f"Current difficulty: {blockchain.difficulty}")
#                     else:
#                         print("⚠ Blockchain verification failed!")
#                         print("The chain may have been tampered with or corrupted.")
                
#                 elif choice == '4':
#                     print("\n=== Saving Blockchain ===")
#                     filename = input("Enter filename (press Enter for default 'blockchain_data.json'): ").strip()
#                     if not filename:
#                         filename = "blockchain_data.json"
#                     blockchain.save_to_file(filename)
#                     print(f"Blockchain saved successfully to {filename}")
                
#                 elif choice == '5':
#                     print("\n=== System Statistics ===")
#                     total_blocks = len(blockchain.chain)
#                     total_recordings = total_blocks - 1  # Excluding genesis block
                    
#                     if total_recordings > 0:
#                         # Collect emotion statistics
#                         emotion_counts = {}
#                         for block in blockchain.chain[1:]:
#                             emotion_counts[block.emotion] = emotion_counts.get(block.emotion, 0) + 1
                        
#                         print(f"Total recordings: {total_recordings}")
#                         print("\nEmotion Distribution:")
#                         for emotion, count in emotion_counts.items():
#                             percentage = (count / total_recordings) * 100
#                             print(f"{emotion}: {count} ({percentage:.1f}%)")
                        
#                         # Show recent activity
#                         print("\nRecent Activity:")
#                         recent_blocks = blockchain.chain[-5:]  # Last 5 blocks
#                         for block in recent_blocks[1:]:  # Skip genesis block if it's in the recent blocks
#                             print(f"Block {block.index}: {block.emotion} - {block.timestamp}")
#                     else:
#                         print("No recordings available yet")
                
#                 elif choice == '6':
#                     print("\nSaving final state...")
#                     blockchain.save_to_file()  # Save before exiting
#                     print("Thank you for using Emotion Recognition Blockchain System!")
#                     break
                
#                 else:
#                     print("\nInvalid choice. Please enter a number between 1 and 6.")
            
#             except KeyboardInterrupt:
#                 print("\n\nOperation cancelled by user")
#                 continue
#             except Exception as e:
#                 logger.error(f"Error in main loop: {str(e)}")
#                 print(f"\nAn error occurred: {str(e)}")
#                 print("Please try again or choose a different option.")
                
#     except Exception as e:
#         logger.error(f"Application startup failed: {str(e)}")
#         print(f"Failed to start application: {str(e)}")
#         print("Please ensure all required dependencies are installed and audio device is properly configured.")

# if __name__ == "__main__":
#     # Configure environment
#     os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
#     # Configure logging format for timestamps
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
    
#     # Print welcome message
#     print("=" * 50)
#     print("Welcome to Emotion Recognition Blockchain System")
#     print("Version 1.0")
#     print("=" * 50)
    
#     # Start the application
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nApplication terminated by user")
#     except Exception as e:
#         print(f"\nFatal error: {str(e)}")
#         print("Application crashed. Please check the logs for details.")
#     finally:
#         print("\nShutting down...")




import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import glob
import json
import base64
import hashlib
import soundfile as sf
import sounddevice as sd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import logging
import io
from typing import Optional, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class AudioProcessingError(Exception):
    pass

class BlockchainError(Exception):
    pass

class ModelError(Exception):
    pass

# Feature Extraction Functions
def extract_feature(file_name=None, audio=None, sample_rate=22050, mfcc=True, chroma=True, mel=True):
    if file_name:
        try:
            with sf.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise AudioProcessingError(f"Failed to load audio file: {e}")
    else:
        X = audio

    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_features))

    if mel:
        mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_features))

    return result

# Emotion Recognition Model
class EmotionRecognizer:
    def __init__(self, model_path: Optional[str] = None):
        self.emotions = ['calm', 'happy', 'fearful', 'disgust']
        self.height = 15
        self.width = 12
        self.channels = 1
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.train_model()

    def train_model(self):
        logger.info("Training new emotion recognition model...")
        try:
            # Load and preprocess data
            x, y = self._load_training_data()
            
            # Split and normalize data
            (x_train, x_test, y_train, y_test), self.label_encoder = self._prepare_data(x, y)
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
            
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise ModelError(f"Failed to train model: {e}")

    def _load_training_data(self):
        x, y = [], []
        emotions = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        dataset_path =  "/Users/sonika.n/Desktop/MINI PROJECT/api/speech-emotion-recognition-ravdess-data/**/*.wav"

        for file in glob.glob(dataset_path):
            file_name = os.path.basename(file)
            emotion = emotions.get(file_name.split("-")[2])
            
            if emotion not in self.emotions:
                continue
                
            feature = extract_feature(file_name=file)
            x.append(feature)
            y.append(emotion)
            
        return np.array(x), y

    def _prepare_data(self, x, y):
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
        
        # Scale features
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        
        # Reshape for CNN
        x_train = x_train.reshape(-1, self.height, self.width, self.channels)
        x_test = x_test.reshape(-1, self.height, self.width, self.channels)
        
        return (x_train, x_test, y_train, y_test), label_encoder

    def _create_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(self.height, self.width, self.channels)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def predict_emotion(self, audio_data: np.ndarray, sample_rate: int) -> str:
        try:
            # Extract features
            features = extract_feature(audio=audio_data, sample_rate=sample_rate)
            features = self.scaler.transform([features])
            features = features.reshape(-1, self.height, self.width, self.channels)
            
            # Predict
            prediction = np.argmax(self.model.predict(features), axis=1)
            predicted_emotion = self.label_encoder.inverse_transform(prediction)[0]
            
            return predicted_emotion
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(f"Failed to predict emotion: {e}")

    def save_model(self, path: str):
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelError(f"Failed to save model: {e}")

    def load_model(self, path: str):
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelError(f"Failed to load model: {e}")

# Blockchain Components
class Block:
    def __init__(self, index: int, timestamp: str, audio_data: str, emotion: str, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.audio_data = audio_data
        self.emotion = emotion
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        try:
            block_string = json.dumps({
                "index": self.index,
                "timestamp": self.timestamp,
                "audio_data": str(self.audio_data),
                "emotion": self.emotion,
                "previous_hash": self.previous_hash,
                "nonce": self.nonce
            }, sort_keys=True)
            return hashlib.sha256(block_string.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            raise BlockchainError("Failed to calculate block hash")

    def mine_block(self, difficulty: int) -> None:
        try:
            while self.hash[:difficulty] != '0' * difficulty:
                self.nonce += 1
                self.hash = self.calculate_hash()
        except Exception as e:
            logger.error(f"Block mining failed: {e}")
            raise BlockchainError("Failed to mine block")

class EmotionAudioBlockchain:
    def __init__(self, difficulty: int = 2):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self) -> None:
        try:
            genesis_block = Block(0, str(datetime.now()), "Genesis Block", "None", "0")
            genesis_block.mine_block(self.difficulty)
            self.chain.append(genesis_block)
        except Exception as e:
            logger.error(f"Genesis block creation failed: {e}")
            raise BlockchainError("Failed to create genesis block")

    def add_block(self, audio_data: str, emotion: str) -> Block:
        try:
            new_block = Block(
                len(self.chain),
                str(datetime.now()),
                audio_data,
                emotion,
                self.chain[-1].hash
            )
            new_block.mine_block(self.difficulty)
            self.chain.append(new_block)
            return new_block
        except Exception as e:
            logger.error(f"Failed to add block: {e}")
            raise BlockchainError("Failed to add new block")

    def is_chain_valid(self) -> bool:
        try:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]

                if current_block.hash != current_block.calculate_hash():
                    return False

                if current_block.previous_hash != previous_block.hash:
                    return False

            return True
        except Exception as e:
            logger.error(f"Chain validation failed: {e}")
            return False

    def get_statistics(self) -> dict:
        try:
            stats = {
                'total_recordings': len(self.chain) - 1,  # Exclude genesis block
                'emotions': {}
            }
            
            for block in self.chain[1:]:  # Skip genesis block
                if block.emotion in stats['emotions']:
                    stats['emotions'][block.emotion] += 1
                else:
                    stats['emotions'][block.emotion] = 1
                    
            return stats
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            raise BlockchainError("Failed to generate statistics")

    def save_to_file(self, filename: str = "blockchain_data.json") -> None:
        try:
            blockchain_data = []
            for block in self.chain:
                block_data = {
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'audio_data': block.audio_data,
                    'emotion': block.emotion,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce
                }
                blockchain_data.append(block_data)

            with open(filename, 'w') as f:
                json.dump(blockchain_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")
            raise BlockchainError("Failed to save blockchain to file")

    def load_from_file(self, filename: str = "blockchain_data.json") -> None:
        try:
            if not os.path.exists(filename):
                logger.info(f"No blockchain file found at {filename}")
                return

            with open(filename, 'r') as f:
                blockchain_data = json.load(f)

            self.chain = []
            for block_data in blockchain_data:
                block = Block(
                    block_data['index'],
                    block_data['timestamp'],
                    block_data['audio_data'],
                    block_data['emotion'],
                    block_data['previous_hash']
                )
                block.hash = block_data['hash']
                block.nonce = block_data['nonce']
                self.chain.append(block)
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            raise BlockchainError("Failed to load blockchain from file")

# Audio Processing Functions
def encode_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Audio encoding failed: {e}")
        raise AudioProcessingError("Failed to encode audio data")

def decode_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
    try:
        audio_bytes = base64.b64decode(audio_base64)
        buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(buffer)
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Audio decoding failed: {e}")
        raise AudioProcessingError("Failed to decode audio data")

def record_and_process(blockchain: EmotionAudioBlockchain, 
                      emotion_recognizer: EmotionRecognizer,
                      duration: int = 5, 
                      sample_rate: int = 22050) -> None:
    try:
        print(f"\nRecording for {duration} seconds...")
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Predict emotion
        emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
        
        # Encode and store in blockchain
        audio_base64 = encode_audio(audio, sample_rate)
        block = blockchain.add_block(audio_base64, emotion)
        
        print(f"\nPredicted Emotion: {emotion}")
        print(f"Block Hash: {block.hash}")
        
        # Save blockchain
        blockchain.save_to_file()
        
    except Exception as e:
        logger.error(f"Recording and processing failed: {e}")
        print(f"An error occurred: {str(e)}")
def play_audio_from_block(block: Block) -> None:
    try:
        if block.audio_data == "Genesis Block":
            print("Cannot play audio from genesis block")
            return
            
        audio_data, sample_rate = decode_audio(block.audio_data)
        sd.play(audio_data, sample_rate)
        sd.wait()
        
    except Exception as e:
        logger.error(f"Failed to play audio: {e}")
        print(f"An error occurred while playing audio: {str(e)}")

def main():
    try:
        print("\nInitializing Emotion Recognition Blockchain System...")
        
        # Initialize blockchain and emotion recognizer
        blockchain = EmotionAudioBlockchain(difficulty=2)
        emotion_recognizer = EmotionRecognizer()  # This will train if no model exists
        
        # Load existing blockchain if available
        blockchain.load_from_file()
        
        while True:
            print("\n=== Emotion Recognition Blockchain System ===")
            print("1. Record new audio")
            print("2. View all recordings")
            print("3. Verify blockchain")
            print("4. View statistics")
            print("5. Exit")
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    record_and_process(blockchain, emotion_recognizer)
                
                elif choice == '2':
                    if len(blockchain.chain) <= 1:
                        print("\nNo recordings found in the blockchain.")
                        continue
                        
                    print("\nStored Recordings:")
                    for block in blockchain.chain[1:]:  # Skip genesis block
                        print(f"\nBlock {block.index}")
                        print(f"Timestamp: {block.timestamp}")
                        print(f"Emotion: {block.emotion}")
                        print(f"Hash: {block.hash}")
                        
                        play_choice = input("Would you like to play this recording? (y/n): ").strip().lower()
                        if play_choice == 'y':
                            print("Playing audio...")
                            play_audio_from_block(block)
                
                elif choice == '3':
                    print("\nVerifying blockchain integrity...")
                    if blockchain.is_chain_valid():
                        print("✅ Blockchain is valid!")
                    else:
                        print("❌ Blockchain validation failed!")
                
                elif choice == '4':
                    stats = blockchain.get_statistics()
                    print("\nBlockchain Statistics:")
                    print(f"Total Recordings: {stats['total_recordings']}")
                    print("\nEmotion Distribution:")
                    for emotion, count in stats['emotions'].items():
                        percentage = (count / stats['total_recordings']) * 100
                        print(f"{emotion}: {count} recordings ({percentage:.1f}%)")
                
                elif choice == '5':
                    print("\nSaving blockchain and exiting...")
                    blockchain.save_to_file()
                    break
                
                else:
                    print("\nInvalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                blockchain.save_to_file()
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"An error occurred: {str(e)}")
                
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        print(f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()



from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

blockchain = EmotionAudioBlockchain(difficulty=2)
emotion_recognizer = EmotionRecognizer()  # Train if no model exists
blockchain.load_from_file()

@app.route("/api/recordings", methods=["GET"])
def get_recordings():
    recordings = [
        {
            "index": block.index,
            "timestamp": block.timestamp,
            "emotion": block.emotion,
            "hash": block.hash
        }
        for block in blockchain.chain[1:]  # Exclude genesis block
    ]
    return jsonify(recordings)

@app.route("/api/predict", methods=["POST"])
def predict_emotion():
    try:
        # Record and process the audio
        duration = 5
        sample_rate = 22050
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        # Predict the emotion
        emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
        audio_base64 = encode_audio(audio, sample_rate)
        block = blockchain.add_block(audio_base64, emotion)
        blockchain.save_to_file()

        return jsonify({"predicted_emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/play/<int:index>", methods=["POST"])
def play_recording(index):
    try:
        if index < 1 or index >= len(blockchain.chain):
            return jsonify({"error": "Invalid block index"}), 400

        block = blockchain.chain[index]
        play_audio_from_block(block)
        return jsonify({"status": "Audio played successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify", methods=["GET"])
def verify_blockchain():
    is_valid = blockchain.is_chain_valid()
    return jsonify({"is_valid": is_valid})

if __name__ == "__main__":
    app.run(debug=True)

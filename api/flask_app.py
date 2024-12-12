# from flask import Flask, jsonify, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for cross-origin requests

# blockchain = EmotionAudioBlockchain(difficulty=2)
# emotion_recognizer = EmotionRecognizer()  # Train if no model exists
# blockchain.load_from_file()

# @app.route("/api/recordings", methods=["GET"])
# def get_recordings():
#     recordings = [
#         {
#             "index": block.index,
#             "timestamp": block.timestamp,
#             "emotion": block.emotion,
#             "hash": block.hash
#         }
#         for block in blockchain.chain[1:]  # Exclude genesis block
#     ]
#     return jsonify(recordings)

# @app.route("/api/predict", methods=["POST"])
# def predict_emotion():
#     try:
#         # Record and process the audio
#         duration = 5
#         sample_rate = 22050
#         audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
#         sd.wait()
#         audio = audio.flatten()

#         # Predict the emotion
#         emotion = emotion_recognizer.predict_emotion(audio, sample_rate)
#         audio_base64 = encode_audio(audio, sample_rate)
#         block = blockchain.add_block(audio_base64, emotion)
#         blockchain.save_to_file()

#         return jsonify({"predicted_emotion": emotion})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/api/play/<int:index>", methods=["POST"])
# def play_recording(index):
#     try:
#         if index < 1 or index >= len(blockchain.chain):
#             return jsonify({"error": "Invalid block index"}), 400

#         block = blockchain.chain[index]
#         play_audio_from_block(block)
#         return jsonify({"status": "Audio played successfully"})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/api/verify", methods=["GET"])
# def verify_blockchain():
#     is_valid = blockchain.is_chain_valid()
#     return jsonify({"is_valid": is_valid})

# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import io
import base64
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import warnings
warnings.filterwarnings('ignore')

# Set page config
# st.set_page_config(
#     page_title="🐾 Animal Sound Translator",
#     page_icon="🐾",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .animal-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .translation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None

@st.cache_resource
def load_models():
    """Load pre-trained models for audio classification"""
    try:
        # Load a general audio classification model
        feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        classifier = pipeline("audio-classification", 
                            model=model, 
                            feature_extractor=feature_extractor)
        return classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def analyze_audio_features(audio_data, sr):
    """Extract comprehensive audio features"""
    features = {}
    
    # Basic features
    features['duration'] = len(audio_data) / sr
    features['sample_rate'] = sr
    features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
    
    # Pitch and tempo
    try:
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        features['avg_pitch'] = np.mean(pitch_values) if pitch_values else 0
        features['pitch_std'] = np.std(pitch_values) if pitch_values else 0
    except:
        features['avg_pitch'] = 0
        features['pitch_std'] = 0
    
    return features

def classify_animal_sound(audio_data, sr, classifier):
    """Classify the animal sound using pre-trained model"""
    try:
        # Resample to 16kHz if needed (common requirement for audio models)
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Ensure audio is not too long (limit to 30 seconds)
        max_length = 30 * sr
        if len(audio_data) > max_length:
            audio_data = audio_data[:max_length]
        
        # Get predictions
        predictions = classifier(audio_data, sampling_rate=sr)
        
        # Filter for animal-related sounds
        animal_keywords = ['dog', 'cat', 'bird', 'cow', 'horse', 'pig', 'sheep', 'goat', 
                          'chicken', 'duck', 'rooster', 'bark', 'meow', 'chirp', 'moo', 
                          'neigh', 'oink', 'baa', 'cluck', 'quack', 'crow', 'howl', 'purr']
        
        animal_predictions = []
        for pred in predictions:
            label_lower = pred['label'].lower()
            if any(keyword in label_lower for keyword in animal_keywords):
                animal_predictions.append(pred)
        
        # If no animal sounds found, return top predictions anyway
        if not animal_predictions:
            animal_predictions = predictions[:3]
        
        return animal_predictions[:5]  # Return top 5
        
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return []

def generate_translation(animal_type, confidence, audio_features):
    """Generate human-readable translation based on animal type and audio features"""
    
    # Animal behavior patterns and translations
    translations = {
        'dog': {
            'high_pitch': "I'm excited! Let's play!",
            'low_pitch': "I'm being protective or warning you.",
            'rapid': "I'm very excited or anxious!",
            'slow': "I'm calm but want your attention.",
            'loud': "I need something urgently!",
            'soft': "I'm content and happy.",
            'default': "Woof! I'm trying to communicate with you!"
        },
        'cat': {
            'high_pitch': "I want something! Feed me or pet me!",
            'low_pitch': "I'm content and relaxed.",
            'rapid': "I'm frustrated or demanding attention!",
            'slow': "I'm greeting you or feeling social.",
            'loud': "I'm upset or in distress!",
            'soft': "I'm happy and comfortable.",
            'default': "Meow! I'm talking to you, human!"
        },
        'bird': {
            'high_pitch': "I'm alerting others or expressing joy!",
            'low_pitch': "I'm establishing territory or calling for a mate.",
            'rapid': "I'm excited or warning of danger!",
            'slow': "I'm content and peaceful.",
            'loud': "I'm calling to my flock or defending my space!",
            'soft': "I'm content and comfortable.",
            'default': "Tweet! I'm singing my song!"
        },
        'cow': {
            'high_pitch': "I'm looking for my calf or feeling distressed!",
            'low_pitch': "I'm calm and content.",
            'loud': "I need attention or I'm calling to the herd!",
            'soft': "I'm peaceful and relaxed.",
            'default': "Moo! I'm communicating with my herd!"
        },
        'default': {
            'high_pitch': "I'm expressing excitement or alertness!",
            'low_pitch': "I'm calm or showing dominance.",
            'rapid': "I'm excited, anxious, or trying to get attention!",
            'slow': "I'm relaxed and content.",
            'loud': "I need attention or I'm expressing strong emotion!",
            'soft': "I'm comfortable and peaceful.",
            'default': "I'm trying to communicate something important!"
        }
    }
    
    # Determine animal category
    animal_key = 'default'
    for key in translations.keys():
        if key in animal_type.lower():
            animal_key = key
            break
    
    # Analyze audio characteristics
    pitch = audio_features.get('avg_pitch', 0)
    energy = audio_features.get('rms_energy', 0)
    zcr = audio_features.get('zero_crossing_rate', 0)
    
    # Determine characteristics
    characteristics = []
    if pitch > 300:
        characteristics.append('high_pitch')
    elif pitch > 0 and pitch < 200:
        characteristics.append('low_pitch')
    
    if energy > 0.1:
        characteristics.append('loud')
    elif energy < 0.05:
        characteristics.append('soft')
    
    if zcr > 0.1:
        characteristics.append('rapid')
    elif zcr < 0.05:
        characteristics.append('slow')
    
    # Get translation
    translation_dict = translations[animal_key]
    translation = translation_dict.get('default', "I'm trying to communicate!")
    
    # Use most specific characteristic available
    for char in characteristics:
        if char in translation_dict:
            translation = translation_dict[char]
            break
    
    # Add confidence-based modifier
    if confidence < 0.3:
        translation = f"[Uncertain] {translation}"
    elif confidence > 0.8:
        translation = f"[Very Confident] {translation}"
    
    return translation

def create_spectrogram(audio_data, sr):
    """Create and return spectrogram plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Waveform
    time = np.linspace(0, len(audio_data)/sr, len(audio_data))
    ax1.plot(time, audio_data, color='#4ECDC4', linewidth=1)
    ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2, cmap='viridis')
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
    plt.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🐾 Animal Sound Translator</h1>', unsafe_allow_html=True)
    st.markdown("### 🐾 Animal sound Translator ###")
    
    # Load models
    with st.spinner("Loading AI models..."):
        classifier = load_models()
    
    if classifier is None:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar
    st.sidebar.header("🎵 Audio Input")
    
    # Audio input options
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio (if supported)"]
    )
    
    audio_file = None
    if input_method == "Upload Audio File":
        audio_file = st.sidebar.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
            help="Upload an audio file containing animal sounds"
        )
    
    # Process audio
    if audio_file is not None:
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            
            # Display audio player
            st.sidebar.audio(audio_file, format='audio/wav')
            
            # Main analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔊 Audio Analysis")
                
                # Create spectrogram
                with st.spinner("Generating spectrogram..."):
                    fig = create_spectrogram(audio_data, sample_rate)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.subheader("📊 Audio Properties")
                
                # Basic audio info
                duration = len(audio_data) / sample_rate
                st.metric("Duration", f"{duration:.2f} seconds")
                st.metric("Sample Rate", f"{sample_rate} Hz")
                st.metric("Channels", "Mono")
                
                # Audio features
                features = analyze_audio_features(audio_data, sample_rate)
                
                st.metric("RMS Energy", f"{features['rms_energy']:.4f}")
                st.metric("Avg Pitch", f"{features['avg_pitch']:.1f} Hz")
                st.metric("Spectral Centroid", f"{features['spectral_centroid']:.1f} Hz")
            
            # Animal Classification
            st.subheader("🐾 Animal Identification")
            
            with st.spinner("Analyzing animal sounds..."):
                predictions = classify_animal_sound(audio_data, sample_rate, classifier)
            
            if predictions:
                # Display top prediction
                top_prediction = predictions[0]
                confidence = top_prediction['score']
                animal_type = top_prediction['label']
                
                # Animal identification card
                st.markdown(f"""
                <div class="animal-card">
                    <h2>🎯 Identified Animal</h2>
                    <h3>{animal_type.title()}</h3>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.subheader("📈 Confidence Levels")
                conf_col1, conf_col2 = st.columns(2)
                
                with conf_col1:
                    # Create confidence chart
                    labels = [pred['label'][:20] + '...' if len(pred['label']) > 20 else pred['label'] 
                             for pred in predictions[:5]]
                    scores = [pred['score'] for pred in predictions[:5]]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    bars = ax.barh(labels, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Top 5 Predictions')
                    ax.set_xlim(0, 1)
                    
                    # Add value labels on bars
                    for i, (bar, score) in enumerate(zip(bars, scores)):
                        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score:.1%}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with conf_col2:
                    # Detailed predictions
                    st.write("**Detailed Predictions:**")
                    for i, pred in enumerate(predictions[:5], 1):
                        st.write(f"{i}. **{pred['label']}** - {pred['score']:.1%}")
                
                # Translation
                st.subheader("🗣️ Translation")
                
                translation = generate_translation(animal_type, confidence, features)
                
                st.markdown(f"""
                <div class="translation-card">
                    <h3>💬 What the animal is saying:</h3>
                    <p style="font-size: 1.2em; font-style: italic;">"{translation}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                st.subheader("🔍 Audio Insights")
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    st.metric("Pitch Variation", f"{features['pitch_std']:.1f} Hz")
                    if features['pitch_std'] > 50:
                        st.write("🎵 High pitch variation - expressive vocalization")
                    else:
                        st.write("🎵 Low pitch variation - steady vocalization")
                
                with insight_col2:
                    st.metric("Zero Crossing Rate", f"{features['zero_crossing_rate']:.3f}")
                    if features['zero_crossing_rate'] > 0.1:
                        st.write("⚡ High activity - excited or agitated")
                    else:
                        st.write("😌 Low activity - calm or relaxed")
                
                with insight_col3:
                    st.metric("Spectral Bandwidth", f"{features['spectral_bandwidth']:.1f} Hz")
                    if features['spectral_bandwidth'] > 2000:
                        st.write("🌈 Rich harmonic content")
                    else:
                        st.write("🎯 Focused frequency content")
                
            else:
                st.warning("No animal sounds detected. Please try uploading a different audio file.")
        
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    
    else:
        # Welcome message
        st.info("👆 Please upload an audio file containing animal sounds to begin analysis!")
        
        # Sample information
        st.subheader("ℹ️ How it works")
        st.write("""
        1. **Upload** an audio file containing animal sounds
        2. **AI Analysis** identifies the animal and analyzes audio features
        3. **Translation** converts the sound into human-readable meaning
        4. **Visualization** shows spectrograms and confidence levels
        
        **Supported animals:** Dogs, cats, birds, cows, horses, pigs, sheep, and more!
        """)
        
        # Technical details
        with st.expander("🔬 Technical Details"):
            st.write("""
            - **Audio Processing:** librosa for feature extraction
            - **AI Model:** Pre-trained audio classification from Hugging Face
            - **Features Analyzed:** MFCC, spectral features, pitch, energy
            - **Translation Logic:** Based on animal behavior patterns and audio characteristics
            - **Confidence Scoring:** Model prediction confidence with uncertainty handling
            """)
if __name__ == "__main__":
    main()
    
# import os
# import torch
# import torchaudio
# from flask import Flask, request, render_template_string
# from transformers import AutoModelForCTC, AutoProcessor

# app = Flask(__name__)

# # Load model and processor
# def load_model():
#     processor = AutoProcessor.from_pretrained("FahriHuseynli/DolphinGemma")
#     model = AutoModelForCTC.from_pretrained("FahriHuseynli/DolphinGemma")
#     return processor, model

# def convert_audio(file_path):
#     """Convert uploaded audio to WAV format at 16kHz mono"""
#     new_path = file_path.rsplit(".", 1)[0] + "_converted.wav"
#     waveform, sample_rate = torchaudio.load(file_path)
#     waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
#     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#     waveform = resampler(waveform)
#     torchaudio.save(new_path, waveform, 16000)
#     return new_path

# def predict_text(model, processor, input_file):
#     speech, sr = torchaudio.load(input_file)
#     input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values
#     with torch.no_grad():
#         logits = model(input_values).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)[0]
#     return transcription

# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string('''
#         <h1>Dolphin Audio to Text</h1>
#         <form action="/predict" method="post" enctype="multipart/form-data">
#             <input type="file" name="audiofile">
#             <input type="submit">
#         </form>
#     ''')

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["audiofile"]
#     if file:
#         path = os.path.join("temp_audio.wav")
#         file.save(path)
#         processor, model = load_model()
#         processed_path = convert_audio(path)
#         text = predict_text(model, processor, processed_path)
#         return f"<h3>Predicted Text:</h3><p>{text}</p>"
#     return "No file uploaded."

# # ✅ Programmatic interface for reuse in other apps
# def analyze_dolphin_audio(audio_file_path: str) -> str:
#     processor, model = load_model()
#     processed_path = convert_audio(audio_file_path)
#     text = predict_text(model, processor, processed_path)
#     return text

# if __name__ == "__main__":
#     app.run(debug=True)


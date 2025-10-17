# 🐾 Animal Sound Translator & Taxonomy Identifier

Welcome to the **AI-Powered Animal Translator & Taxonomy Identifier**, a dual-module project that brings together AI, audio/image processing, and open data APIs to help humans better understand and identify animals — through both sound and sight.

## 🚀 Overview

Animals communicate in many fascinating ways — through vocalizations and visual cues. Our project is split into two main modules:

1. **🐾 Animal Sound Translator** — Analyzes animal audio and generates behavior-based translations.
2. **🧬 Animal Image Identifier** — Classifies animals from images and retrieves their full taxonomy.

## Features

- **Real-time Audio Analysis**: Upload audio files and get instant analysis
- **AI-Powered Classification**: Uses pre-trained models from Hugging Face to identify animals
- **Visual Analysis**: 
  - Audio waveform visualization
  - Detailed spectrograms
  - Confidence level charts
- **Smart Translation**: Converts animal sounds to human language based on:
  - Audio characteristics (pitch, energy, frequency)
  - Animal behavior patterns
  - Confidence levels
- **Comprehensive Metrics**: 
  - Audio features (MFCC, spectral analysis)
  - Pitch analysis and variation
  - Energy levels and patterns

  **🐯 Animal Image Identifier**
- **Image classification** using `ResNet-50` via Hugging Face
- **Taxonomy retrieval** using Wikipedia & Wikidata APIs
- **Full hierarchy extraction** from species to kingdom
- Clean UI built in `Streamlit`

## Supported Animals

- Dogs (barking, whining, howling)
- Cats (meowing, purring)
- Birds (chirping, singing)
- Cows (mooing)
- Horses (neighing)
- Pigs (oinking)
- Sheep (bleating)
- And many more!


## 🛠️ Technologies Used

| Category         | Tools/Tech                      |
|------------------|----------------------------------|
| Language         | Python 3.8+                      |
| Web Framework    | Streamlit                        |
| AI Models        | Hugging Face (ResNet-50, Audio)  |
| Signal Processing| librosa                          |
| APIs             | Wikipedia API, Wikidata API      |

---

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
```bash
python -m pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run echoanimal.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`)

3. Upload an audio file containing animal sounds

4. View the analysis results:
   - Animal identification with confidence scores
   - Audio visualizations (waveform and spectrogram)
   - Human translation of the animal communication
   - Detailed audio insights and metrics

## Technical Details

### AI Models
- Uses pre-trained audio classification models from Hugging Face
- Specifically optimized for animal sound recognition
- Confidence scoring with uncertainty handling

### Audio Processing
- **librosa**: Advanced audio analysis and feature extraction
- **MFCC Analysis**: Mel-frequency cepstral coefficients for sound characterization
- **Spectral Features**: Centroid, bandwidth, rolloff analysis
- **Pitch Detection**: Fundamental frequency analysis
- **Energy Analysis**: RMS energy and zero-crossing rate

### Translation Logic
- Based on scientific animal behavior research
- Considers audio characteristics:
  - Pitch (high/low frequency content)
  - Energy (loud/soft intensity)
  - Temporal patterns (rapid/slow variations)
- Contextual interpretation based on animal species
- Confidence-weighted translations

## File Formats Supported

- WAV
- MP3
- OGG
- FLAC
- M4A

## Technical Approach — Taxonomy Identification 🧠
 1.**Image Classification**:Hugging Face pipeline with microsoft/resnet-50

 2.**Wikipedia API**:Correct species title retrieval

 3.**Wikidata API**:Recursive search to fetch complete taxonomy levels
 
 4.Automated extraction up to Kingdom level

## Limitations

- Translation is based on general animal behavior patterns, not actual animal language
- Accuracy depends on audio quality and clarity
- Best results with clear, isolated animal sounds
- Some exotic animals may not be recognized

## Future Enhancements

- Real-time microphone input
- Multi-animal sound separation
- Emotional state detection
- Extended animal database
- Mobile app version

## Contributing

Feel free to contribute by:
- Adding support for more animals
- Improving translation accuracy
- Enhancing the user interface
- Adding new audio analysis features

---

## 🧠 Features

### 🔊 Animal Sound Translator
- **Real-time audio analysis** using `librosa` & `Hugging Face` audio models
- **Feature extraction**: MFCCs, pitch, energy, spectral features
- **Behavior translation**: Maps sound patterns to behaviors using science-backed logic
- **Visualizations**: Audio waveform, spectrograms, confidence scores

---

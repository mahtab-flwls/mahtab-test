# 🎵 Voice Note Speech/Singing Separator 🎤

An intelligent audio processing application that can automatically separate speech from singing in voice notes and audio recordings using machine learning techniques.

## ✨ Features

- **🔍 Audio Analysis**: Analyze audio files to identify speech vs singing segments
- **✂️ Smart Separation**: Automatically separate speech and singing into different audio files
- **🎯 Machine Learning**: Uses advanced audio feature extraction and classification
- **🖥️ Dual Interface**: Both GUI and command-line interfaces available
- **📊 Visualization**: Visual timeline and distribution charts of audio content
- **🏋️ Custom Training**: Train your own classifier with custom audio samples
- **🎵 Multiple Formats**: Supports WAV, MP3, FLAC, M4A, and OGG audio formats

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd voice-note-separator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI application**:
   ```bash
   python gui_app.py
   ```

4. **Or use the command-line interface**:
   ```bash
   python cli_app.py --help
   ```

## 🖥️ GUI Application

The GUI provides an intuitive interface for audio processing:

### Features:
- **File Selection**: Easy audio file selection with drag-and-drop support
- **Real-time Analysis**: Progress tracking with visual feedback
- **Results Visualization**: Timeline charts and pie charts showing speech/singing distribution
- **Batch Processing**: Process multiple files efficiently
- **Export Options**: Choose output directory and format

### Usage:
1. Launch the GUI: `python gui_app.py`
2. Click "Select Audio File" to choose your voice note
3. Click "Analyze Audio" to see the speech/singing breakdown
4. Click "Separate Speech & Singing" to create separate audio files
5. Choose output directory and view results

## 💻 Command-Line Interface

For automation and batch processing, use the CLI:

### Commands:

#### Analyze Audio
```bash
# Basic analysis
python cli_app.py analyze input.wav

# Detailed analysis with segment breakdown
python cli_app.py analyze input.wav --verbose
```

#### Separate Audio
```bash
# Basic separation
python cli_app.py separate input.wav -o output_folder

# Separation with analysis
python cli_app.py separate input.wav -o output_folder --analyze
```

#### Train Custom Classifier
```bash
# Train with your own data
python cli_app.py train --speech-dir speech_samples --singing-dir singing_samples
```

## 🧠 How It Works

### Audio Feature Extraction
The system extracts multiple audio features to distinguish between speech and singing:

- **Spectral Features**: Centroid, rolloff, bandwidth
- **Temporal Features**: Zero crossing rate, RMS energy
- **Harmonic Features**: MFCC coefficients, chroma features
- **Rhythmic Features**: Tempo detection

### Classification Algorithm
- **Default Classifier**: Rule-based heuristics for immediate use
- **Machine Learning**: Random Forest classifier with feature scaling
- **Custom Training**: Train with your own speech/singing samples

### Audio Processing Pipeline
1. **Load Audio**: Convert to standard format (22kHz mono)
2. **Segmentation**: Split into 2-second analysis windows
3. **Feature Extraction**: Extract 37 audio features per segment
4. **Classification**: Predict speech vs singing for each segment
5. **Reconstruction**: Combine segments into separate output files

## 📊 Technical Details

### Supported Audio Formats
- **WAV** (recommended for best quality)
- **MP3** (most common format)
- **FLAC** (lossless compression)
- **M4A** (Apple format)
- **OGG** (open source format)

### System Requirements
- **Python**: 3.7 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 1GB free space for dependencies
- **OS**: Windows, macOS, or Linux

### Dependencies
- **librosa**: Audio analysis and feature extraction
- **scikit-learn**: Machine learning algorithms
- **numpy/scipy**: Numerical computing
- **matplotlib**: Visualization
- **tkinter**: GUI framework
- **soundfile**: Audio I/O

## 🎯 Use Cases

### Personal Use
- **Voice Notes**: Separate speech from background singing
- **Interviews**: Extract clear speech from noisy recordings
- **Lectures**: Remove musical interludes from educational content

### Professional Use
- **Content Creation**: Prepare audio for podcasts and videos
- **Music Production**: Isolate vocal tracks for remixing
- **Research**: Analyze speech patterns in musical contexts

### Educational Use
- **Language Learning**: Focus on speech without musical distractions
- **Music Analysis**: Study the relationship between speech and singing
- **Audio Engineering**: Learn about audio classification techniques

## 🔧 Advanced Usage

### Custom Classifier Training

To improve accuracy for your specific use case:

1. **Prepare Training Data**:
   ```
   training_data/
   ├── speech_samples/
   │   ├── speech1.wav
   │   ├── speech2.wav
   │   └── ...
   └── singing_samples/
       ├── singing1.wav
       ├── singing2.wav
       └── ...
   ```

2. **Train Classifier**:
   ```bash
   python cli_app.py train --speech-dir training_data/speech_samples --singing-dir training_data/singing_samples
   ```

3. **Use Trained Model**: The trained model will automatically be used for future classifications

### Batch Processing Script

Create a script for processing multiple files:

```python
from audio_separator import AudioSeparator
import os

separator = AudioSeparator()
input_dir = "voice_notes"
output_dir = "separated_audio"

for filename in os.listdir(input_dir):
    if filename.endswith(('.wav', '.mp3')):
        input_path = os.path.join(input_dir, filename)
        separator.separate_audio(input_path, output_dir)
```

## 🐛 Troubleshooting

### Common Issues

**Installation Problems**:
- Ensure Python 3.7+ is installed
- Use `pip install --upgrade pip` before installing dependencies
- On macOS, you might need: `brew install portaudio`

**Audio Loading Errors**:
- Check if the audio file is corrupted
- Try converting to WAV format first
- Ensure file path doesn't contain special characters

**Memory Issues**:
- Process shorter audio files (< 10 minutes)
- Close other applications to free up RAM
- Use the CLI for better memory management

**Classification Accuracy**:
- Train a custom classifier with your specific audio types
- Ensure training data is clean and well-labeled
- Use high-quality audio files for better results

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **librosa** team for excellent audio processing tools
- **scikit-learn** community for machine learning algorithms
- **tkinter** developers for GUI framework
- Audio processing research community for foundational work

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the troubleshooting section
- Review existing issues for solutions

---

**Made with ❤️ for the audio processing community**
#!/usr/bin/env python3
"""
Command-line interface for the Voice Note Speech/Singing Separator
"""

import argparse
import os
import sys
from audio_separator import AudioSeparator

def print_banner():
    """Print application banner."""
    banner = """
🎵 Voice Note Speech/Singing Separator 🎤
=========================================
Separate speech from singing in audio files using machine learning.
    """
    print(banner)

def analyze_command(args):
    """Handle analyze command."""
    separator = AudioSeparator()
    
    print(f"🔍 Analyzing audio file: {args.input}")
    
    try:
        results = separator.analyze_audio(args.input)
        
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"📁 File: {os.path.basename(args.input)}")
        print(f"⏱️ Total Duration: {results['total_duration']:.2f} seconds")
        print(f"🗣️ Speech Duration: {results['speech_duration']:.2f} seconds ({results['speech_percentage']:.1f}%)")
        print(f"🎵 Singing Duration: {results['singing_duration']:.2f} seconds ({results['singing_percentage']:.1f}%)")
        
        if args.verbose:
            print(f"\n📋 SEGMENT BREAKDOWN:")
            print("=" * 50)
            for i, (start, end, label) in enumerate(results['segments']):
                start_sec = start / separator.sample_rate
                end_sec = end / separator.sample_rate
                duration = end_sec - start_sec
                
                icon = "🗣️" if label == "speech" else "🎵"
                print(f"{icon} Segment {i+1}: {start_sec:.1f}s - {end_sec:.1f}s ({duration:.1f}s) - {label.upper()}")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

def separate_command(args):
    """Handle separate command."""
    separator = AudioSeparator()
    
    print(f"✂️ Separating speech and singing from: {args.input}")
    
    try:
        speech_path, singing_path = separator.separate_audio(args.input, args.output)
        
        print(f"\n✅ SEPARATION COMPLETE!")
        print("=" * 50)
        print(f"📁 Output Directory: {args.output}")
        print(f"🗣️ Speech File: {speech_path}")
        print(f"🎵 Singing File: {singing_path}")
        
        # Check file sizes
        speech_size = os.path.getsize(speech_path) if os.path.exists(speech_path) else 0
        singing_size = os.path.getsize(singing_path) if os.path.exists(singing_path) else 0
        
        print(f"\n📊 File Sizes:")
        print(f"🗣️ Speech: {speech_size / 1024:.1f} KB")
        print(f"🎵 Singing: {singing_size / 1024:.1f} KB")
        
        if args.analyze:
            print(f"\n🔍 Running analysis...")
            results = separator.analyze_audio(args.input)
            print(f"📊 Speech: {results['speech_percentage']:.1f}% | Singing: {results['singing_percentage']:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during separation: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

def train_command(args):
    """Handle train command."""
    separator = AudioSeparator()
    
    print(f"🏋️ Training classifier...")
    print(f"📁 Speech files directory: {args.speech_dir}")
    print(f"📁 Singing files directory: {args.singing_dir}")
    
    try:
        # Get all audio files from directories
        speech_files = []
        singing_files = []
        
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
        
        if os.path.isdir(args.speech_dir):
            for file in os.listdir(args.speech_dir):
                if file.lower().endswith(audio_extensions):
                    speech_files.append(os.path.join(args.speech_dir, file))
        
        if os.path.isdir(args.singing_dir):
            for file in os.listdir(args.singing_dir):
                if file.lower().endswith(audio_extensions):
                    singing_files.append(os.path.join(args.singing_dir, file))
        
        if not speech_files or not singing_files:
            print("❌ Error: No audio files found in one or both directories.")
            return 1
        
        print(f"📊 Found {len(speech_files)} speech files and {len(singing_files)} singing files")
        
        separator.train_classifier(speech_files, singing_files)
        
        print("✅ Training complete! Classifier saved.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Voice Note Speech/Singing Separator - Separate speech from singing in audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze an audio file
  python cli_app.py analyze input.wav
  
  # Separate speech and singing
  python cli_app.py separate input.wav -o output_folder
  
  # Separate with analysis
  python cli_app.py separate input.wav -o output_folder --analyze
  
  # Train custom classifier
  python cli_app.py train --speech-dir speech_samples --singing-dir singing_samples
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze audio file for speech/singing content')
    analyze_parser.add_argument('input', help='Input audio file path')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed segment breakdown')
    
    # Separate command
    separate_parser = subparsers.add_parser('separate', help='Separate speech and singing into different files')
    separate_parser.add_argument('input', help='Input audio file path')
    separate_parser.add_argument('-o', '--output', default='output', help='Output directory (default: output)')
    separate_parser.add_argument('--analyze', action='store_true', help='Also show analysis results')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train custom classifier with your own data')
    train_parser.add_argument('--speech-dir', required=True, help='Directory containing speech audio files')
    train_parser.add_argument('--singing-dir', required=True, help='Directory containing singing audio files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        return 1
    
    # Print banner for all commands
    print_banner()
    
    # Validate input file exists for analyze and separate commands
    if args.command in ['analyze', 'separate']:
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' does not exist.", file=sys.stderr)
            return 1
    
    # Validate directories for train command
    if args.command == 'train':
        if not os.path.isdir(args.speech_dir):
            print(f"❌ Error: Speech directory '{args.speech_dir}' does not exist.", file=sys.stderr)
            return 1
        if not os.path.isdir(args.singing_dir):
            print(f"❌ Error: Singing directory '{args.singing_dir}' does not exist.", file=sys.stderr)
            return 1
    
    # Execute command
    if args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'separate':
        return separate_command(args)
    elif args.command == 'train':
        return train_command(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
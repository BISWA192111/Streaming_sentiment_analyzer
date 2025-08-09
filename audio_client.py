#!/usr/bin/env python3
import socket
import numpy as np
import soundfile as sf
import io
import time
import sys

HOST = 'localhost'
PORT = 43007
SAMPLING_RATE = 16000

def send_audio_file(audio_file_path):
    """Send an audio file to the Whisper server for transcription"""
    try:
        # Load audio file
        print(f"Loading audio file: {audio_file_path}")
        audio, sr = sf.read(audio_file_path)
        
        # Resample to 16kHz if needed
        if sr != SAMPLING_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLING_RATE}Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Convert to int16 for transmission
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Connect to server
        print(f"Connecting to {HOST}:{PORT}...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("Connected! Sending audio data...")
            
            # Send audio in chunks
            chunk_size = 1024  # samples per chunk
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                s.sendall(chunk.tobytes())
                time.sleep(0.1)  # Small delay to simulate real-time
                
                # Try to receive transcription
                s.settimeout(0.1)
                try:
                    response = s.recv(1024)
                    if response:
                        print(f"Transcription: {response.decode()}")
                except socket.timeout:
                    pass
                s.settimeout(None)
            
            print("Audio sent successfully!")
            
            # Wait for final responses
            time.sleep(2)
            try:
                while True:
                    response = s.recv(1024)
                    if not response:
                        break
                    print(f"Final transcription: {response.decode()}")
            except:
                pass
                
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
    except Exception as e:
        print(f"Error: {e}")

def test_connection():
    """Test basic connection to the server"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("✓ Connection successful!")
            return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        send_audio_file(audio_file)
    else:
        print("Testing connection...")
        if test_connection():
            print("\nUsage:")
            print(f"python {sys.argv[0]} <audio_file.wav>")
            print("\nExample:")
            print(f"python {sys.argv[0]} test_audio.wav")

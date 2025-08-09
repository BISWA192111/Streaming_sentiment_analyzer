#!/usr/bin/env python3
"""
Quick microphone volume test tool
Helps you adjust microphone settings before using live transcription
"""
import pyaudio
import numpy as np
import time
import sys

SAMPLING_RATE = 16000
CHUNK_SIZE = 1600
CHANNELS = 1
FORMAT = pyaudio.paInt16

def test_microphone_levels():
    """Test microphone and show real-time audio levels"""
    print("🎙️ Microphone Volume Test")
    print("=" * 30)
    print("This will help you adjust your microphone volume.")
    print("Speak into your microphone and watch the levels below:")
    print("")
    print("📊 Level Guide:")
    print("   0-50:   Too quiet - increase microphone volume")
    print("   50-200: Good for speech recognition")
    print("   200+:   Very loud - may be too loud")
    print("")
    print("Press Ctrl+C to stop\n")
    
    p = pyaudio.PyAudio()
    
    try:
        # List devices
        print("Available microphones:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']}")
        
        choice = input("\nPress Enter for default, or type device number: ").strip()
        
        if choice.isdigit():
            device_index = int(choice)
        else:
            device_index = None
        
        # Open stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLING_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("\n🎤 Listening... (speak now)")
        print("=" * 50)
        
        max_level = 0
        samples = 0
        
        while True:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio_data).mean()
                
                max_level = max(max_level, level)
                samples += 1
                
                # Create visual bar
                bar_length = min(int(level / 10), 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                status = ""
                if level < 50:
                    status = "TOO QUIET ❌"
                elif level < 200:
                    status = "GOOD ✅"
                else:
                    status = "VERY LOUD ⚠️"
                
                print(f"\r📊 {level:6.1f} |{bar}| {status}     ", end="", flush=True)
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
        
        print(f"\n\n📈 Test Results:")
        print(f"   Maximum level detected: {max_level:.1f}")
        print(f"   Samples tested: {samples}")
        
        if max_level < 50:
            print(f"\n⚠️ Your microphone is too quiet!")
            print(f"   Please increase microphone volume in Windows:")
            print(f"   1. Right-click speaker icon → Open Sound settings")
            print(f"   2. Input section → Device properties")
            print(f"   3. Increase volume to 80-100%")
            print(f"   4. Enable microphone boost if available")
        elif max_level > 500:
            print(f"\n⚠️ Your microphone might be too loud!")
            print(f"   Consider reducing microphone volume slightly.")
        else:
            print(f"\n✅ Microphone volume looks good!")
            print(f"   You should be ready for live transcription.")
        
        stream.close()
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    test_microphone_levels()

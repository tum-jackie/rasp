import numpy as np
import timeit
import sounddevice as sd
import scipy.signal
import python_speech_features
from tflite_runtime.interpreter import Interpreter
import soundfile as sf

# Parameters
debug_time = 1
debug_acc = 0
word_threshold = 0.5
rec_duration = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = '/content/oxpecker1.tflite'
audio_file_path = '/content/2XC84142 - Red-billed Oxpecker - Buphagus erythrorynchus1.wav'  # Replace with the path to your audio file

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read audio file
audio_data, original_sample_rate = sf.read(audio_file_path)

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs


# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features


    mfccs = python_speech_features.base.mfcc(audio_data,
                                          samplerate=resample_rate,
                                          winlen=0.256,
                                          winstep=0.050,
                                          numcep=num_mfcc,
                                          nfilt=26,
                                          nfft=2048,
                                          preemph=0.0,
                                          ceplifter=0,
                                          appendEnergy=False,
                                          winfunc=np.hanning)
    mfccs = mfccs.transpose()
# Ensure the desired shape is (1, 16, 16, 1)
    desired_shape = (16, 16)


# Check if the current shape is different and reshape if needed
    if mfccs.shape != desired_shape:
        mfccs = np.resize(mfccs, desired_shape)
    mfccs = np.resize(mfccs, desired_shape)
# Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data.argmax()
    print(prediction)
    val = output_data[0][0]
    if val > word_threshold:
        print('stop')
    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
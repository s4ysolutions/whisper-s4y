import tensorflow as tf
from transformers import WhisperProcessor
import tensorflow_io as tfio
from settings import model_name, tflite_model_path
import time

# huggingface utility to prepare audio data for input and
# decode output tokens to redable string
processor = WhisperProcessor.from_pretrained(model_name)

# A helper function to extract array of raw PCM floats from wav file
def wav_audio(wav_file_path):
    waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(wav_file_path))
    if sample_rate != 16000:
        print(f"sample rate is {sample_rate}, resampling...")
        waveform = tfio.audio.resample(waveform, rate_in=sample_rate, rate_out=16000)
        print("ok")
    audio = waveform[:, 0]
    return audio

def test():
    # model check
    # read a "waveform" - an array of the floats forming a voice raw data
    audio = wav_audio('al-fatiha.wav')
    # we need to conver wave form to "mel spectrogram"
    # namely to turn the 1-dimenstion array of PCM values
    # to n-dimension array of the set of frequneces for very small duration of
    # audio record which being summed restore PCM value at that time
    # (Fourier transform if such term is easier)
    inputs = processor(audio, sampling_rate=16000, return_tensors="tf")
    input_features = inputs.input_features

    # this commented out code for testing the original models:
    # just call their `generate` method
    #model = TFWhisperForConditionalGeneration.from_pretrained(model_name)
    #generated_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)
    #model = GenerateModel(model=model, forced_decoder_ids=forced_decoder_ids)
    #generated_ids = model.generate(input_features)

    # our purpose is to check the Lite model
    interpreter = tf.lite.Interpreter(tflite_model_path)
    # This magic call give us the `serving` method of the wrapper class
    # If we did not have the only method we would have had to use more complex approach
    # to find the needed serving. Google for get_signature_runner if interested
    tflite_generate = interpreter.get_signature_runner()
    # a workhorse: calls the serving which in turn will call `generate` method
    # of the original model
    start_time = time.time()
    output = tflite_generate(input_features=input_features)
    end_time = time.time()
    generated_ids = output["sequences"]
    # now we have an array of `tokens` - integer values
    # and need to convert them to the string
    # use huggingface utility class to do so
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"transcription tflite: {transcription}")
    duration = end_time - start_time
    print(f"Duration: {duration:.4f} seconds")

if __name__ == "__main__":
    test()

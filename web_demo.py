from transformers import AutoModel, AutoTokenizer, AutoConfig
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import random
import hashlib
import requests
import numpy as np
import warnings
import gradio as gr
import base64
from io import BytesIO
import mdtex2html
import argparse
import tempfile
import torch
import wavio
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./base-model/chatglm-6b", type=str)
parser.add_argument(
    "--pt_path", default="./child-emotional-code/checkpoints/checkpoint-3000", type=str
)
args = parser.parse_args()


def translate(query):
    # Set your own appid/appkey.
    appid = "20231010001842773"
    appkey = "yTa0u9dzOJkdlheuU9Lo"

    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = "auto"
    to_lang = "yue"

    endpoint = "http://api.fanyi.baidu.com"
    path = "/api/trans/vip/translate"
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding="utf-8"):
        return hashlib.md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "appid": appid,
        "q": query,
        "from": from_lang,
        "to": to_lang,
        "salt": salt,
        "sign": sign,
    }

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    # Show response
    translated_text = result["trans_result"][0]["dst"]

    file_name = tempfile.mktemp(suffix=".wav", prefix="audio_output_", dir="./")
    speech_config = speechsdk.SpeechConfig(
        subscription="1af60babed7c49e9974354c9d82df939", region="eastus"
    )
    speech_config.speech_synthesis_language = "zh-HK"
    speech_config.speech_synthesis_voice_name = "zh-HK-HiuMaanNeural"
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=None
    )
    speech_synthesizer.speak_text_async(translated_text)
    result = speech_synthesizer.speak_text_async(translated_text).get()
    stream = speechsdk.AudioDataStream(result)
    stream.save_to_wav_file(file_name)

    wave_file = wavio.read(file_name)
    audio_data = wave_file.data
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))
    rate = wave_file.rate
    sampwidth = wave_file.sampwidth
    audio_bytes = BytesIO()
    wavio.write(audio_bytes, audio_data, rate, sampwidth=sampwidth)
    audio_bytes.seek(0)

    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = (
        f'<audio src="data:audio/mpeg;base64,{audio_base64}" controls autoplay></audio>'
    )
    os.remove(file_name)
    return translated_text, audio_player


def initialize_model(model_path, p_tuning_path=""):
    if p_tuning_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, pre_seq_len=128
        )
        config.prefix_n_experts = 2
        config.prefix_cur_expert = -1
        config.expert_weights = [0.0, 1.0]

        model = AutoModel.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )

        # 此处使用你的 ptuning 工作目录
        prefix_state_dict = torch.load(
            os.path.join(p_tuning_path, "pytorch_model.bin",)
        )
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            new_prefix_state_dict[k[len("transformer.prefix_encoder.") :]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        # V100 机型上可以不进行量化
        # print(f"Quantized to 4 bit")
        model = model.quantize(8)
        model = model.half().cuda()
        model.transformer.prefix_encoder.float()
        model = model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        )
        model = model.eval()
    return tokenizer, model


tokenizer, model = initialize_model(args.model_path, args.pt_path)


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(audio, chatbot, max_length, top_p, temperature, history):
    input = transcribe(audio)
    print(f">>>>>>>>>> input: {input}")
    chatbot.append((parse_text(input), ""))
    response, history = model.chat(
        tokenizer,
        input,
        history=history,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
    )
    response, audio_player = translate(response)
    print(f">>>>>>>> response: {response}")
    chatbot[-1] = (parse_text(input), parse_text(response))
    return chatbot, history, audio_player
    # for response, history in model.stream_chat(
    #     tokenizer,
    #     input,
    #     history,
    #     max_length=max_length,
    #     top_p=top_p,
    #     temperature=temperature,
    # ):
    #     chatbot[-1] = (parse_text(input), parse_text(response))

    #     yield chatbot, history


def transcribe(raw_audio):
    sr, y = raw_audio
    data = convert_to_16_bit_wav(y)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sr,
        sample_width=data.dtype.itemsize,
        channels=(1 if len(data.shape) == 1 else data.shape[1]),
    )
    file_name = tempfile.mktemp(suffix=".wav", prefix="audio_input_", dir="./")
    file = audio.export(file_name, format="wav")
    file.close()
    speech_config = speechsdk.SpeechConfig(
        subscription="1af60babed7c49e9974354c9d82df939", region="eastus"
    )
    speech_config.speech_recognition_language = "zh-HK"

    audio_config = speechsdk.audio.AudioConfig(filename=file_name)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print(
            "No speech could be recognized: {}".format(
                speech_recognition_result.no_match_details
            )
        )
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    os.remove(file_name)
    return speech_recognition_result.text


def reset_user_input():
    return gr.update(value=None)


def reset_state():
    return [], []


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        warnings.warn(warning.format(data.dtype))
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(warning.format(data.dtype))
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        warnings.warn(warning.format(data.dtype))
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        warnings.warn(warning.format(data.dtype))
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            # with gr.Column(scale=12):
            #     user_input = gr.Textbox(
            #         show_label=False, placeholder="Input...", lines=10
            #     ).style(container=False)
            with gr.Column(scale=1):
                audio_input = gr.Microphone()
                audio_output = gr.HTML(visible=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True
            )
            top_p = gr.Slider(
                0, 1, value=0.7, step=0.01, label="Top P", interactive=True
            )
            temperature = gr.Slider(
                0, 1, value=0.95, step=0.01, label="Temperature", interactive=True
            )

    history = gr.State([])

    # submitBtn.click(transcribe, [audio_input], [user_input])
    submitBtn.click(
        predict,
        [audio_input, chatbot, max_length, top_p, temperature, history],
        [chatbot, history, audio_output],
        show_progress=True,
    )
    submitBtn.click(reset_user_input, [], [audio_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0")

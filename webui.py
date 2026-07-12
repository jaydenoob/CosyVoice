# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by zhuzehao in 2026:
# - Added local WebUI improvements for CosyVoice3 0.5B usage.
# - Added segmented generation preview, progress, cancellation request,
#   completion notification, and saved WAV output for browser playback.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import soundfile as sf
import random
import librosa
import uuid
import queue
import threading
import time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'webui_outputs')
NOTIFICATION_AUDIO = os.path.join(OUTPUT_DIR, 'notification_done.wav')
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                 '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                 '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                 '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}
stream_mode_list = [('否', False), ('是', True)]
max_val = 0.8
MODEL_IS_COSYVOICE3 = False
MODEL_NAME = 'CosyVoice'
PARTIAL_PREVIEW_PAUSE_SECONDS = 1.5
generation_state_lock = threading.Lock()
generation_state = {
    'active': False,
    'cancel_event': None,
}


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def ensure_endofprompt(text):
    token = '<|endofprompt|>'
    text = text.strip()
    if text == '':
        return text
    if token in text:
        return text
    return text + token


def ensure_cosyvoice3_cross_lingual_text(text):
    token = '<|endofprompt|>'
    prefix = 'You are a helpful assistant.<|endofprompt|>'
    text = text.strip()
    if text == '':
        return text
    if token in text:
        return text
    return prefix + text


def save_audio(audio_data, sample_rate):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audio_data = np.asarray(audio_data).flatten()
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = np.max(np.abs(audio_data)) if audio_data.size > 0 else 0
    if max_abs > 1.0:
        audio_data = audio_data / max_abs
    audio_data = np.clip(audio_data, -1.0, 1.0)
    output_path = os.path.join(OUTPUT_DIR, 'tts_{}.wav'.format(uuid.uuid4().hex))
    sf.write(output_path, audio_data, sample_rate, subtype='PCM_16')
    return output_path


def prepare_prompt_wav(prompt_wav):
    if prompt_wav is None:
        return None, None, None
    try:
        return prompt_wav, torchaudio.info(prompt_wav), None
    except Exception as first_error:
        try:
            audio_data, sample_rate = librosa.load(prompt_wav, sr=None, mono=True)
            if audio_data.size == 0:
                return None, None, 'prompt音频为空或无法解码'
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            converted_path = os.path.join(OUTPUT_DIR, 'prompt_{}.wav'.format(uuid.uuid4().hex))
            sf.write(converted_path, np.clip(audio_data, -1.0, 1.0), sample_rate, subtype='PCM_16')
            return converted_path, torchaudio.info(converted_path), None
        except Exception as second_error:
            logging.warning('failed to read prompt audio {}, torchaudio error: {}, librosa error: {}'.format(
                prompt_wav, first_error, second_error))
            return None, None, 'prompt音频读取失败，请换成 wav/flac/mp3，或重新上传/录制'


def ensure_notification_audio():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(NOTIFICATION_AUDIO):
        return NOTIFICATION_AUDIO
    sample_rate = 24000
    tone_1 = np.sin(2 * np.pi * 880 * np.linspace(0, 0.16, int(sample_rate * 0.16), endpoint=False))
    silence = np.zeros(int(sample_rate * 0.04))
    tone_2 = np.sin(2 * np.pi * 1175 * np.linspace(0, 0.18, int(sample_rate * 0.18), endpoint=False))
    envelope_1 = np.hanning(tone_1.size * 2)[:tone_1.size]
    envelope_2 = np.hanning(tone_2.size * 2)[:tone_2.size]
    audio = np.concatenate([tone_1 * envelope_1, silence, tone_2 * envelope_2]) * 0.35
    sf.write(NOTIFICATION_AUDIO, audio, sample_rate, subtype='PCM_16')
    return NOTIFICATION_AUDIO


def progress_text(completed, total, start_time, estimate=None):
    elapsed = time.time() - start_time
    parts = [
        '已完成 {}/{} 段'.format(min(completed, total), total),
        '已运行 {}'.format(format_duration(elapsed)),
    ]
    if estimate is not None and completed < total:
        parts.append('预计剩余 {}'.format(format_duration(estimate['remaining_seconds'])))
        parts.append('预计 {} 完成'.format(estimate['finish_time_text']))
    return '，'.join(parts)


def cancel_progress_text(completed, total, start_time):
    elapsed = time.time() - start_time
    return '已请求中断生成，已完成 {}/{} 段，已运行 {}'.format(
        min(completed, total), total, format_duration(elapsed))


def format_duration(seconds):
    seconds = max(0.0, seconds)
    if seconds < 60:
        return '{:.1f}s'.format(seconds)
    return '{:.1f}s（{:.1f}分钟）'.format(seconds, seconds / 60)


def update_estimate(completed, total, start_time):
    if completed <= 0 or completed >= total:
        return None
    elapsed = time.time() - start_time
    average_seconds = elapsed / completed
    remaining_seconds = average_seconds * (total - completed)
    finish_time = time.localtime(time.time() + remaining_seconds)
    return {
        'remaining_seconds': remaining_seconds,
        'finish_time_text': time.strftime('%H:%M', finish_time),
    }


def begin_generation():
    with generation_state_lock:
        if generation_state['active']:
            return None
        cancel_event = threading.Event()
        generation_state['active'] = True
        generation_state['cancel_event'] = cancel_event
        return cancel_event


def finish_generation(cancel_event):
    with generation_state_lock:
        if generation_state['cancel_event'] is cancel_event:
            generation_state['active'] = False
            generation_state['cancel_event'] = None


def request_cancel_generation():
    with generation_state_lock:
        cancel_event = generation_state['cancel_event']
    if cancel_event is None:
        return '当前没有正在生成的任务'
    cancel_event.set()
    return '已请求中断生成，当前段可能需要等模型返回后才会完全停止'


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    def fail(message):
        logging.warning(message)
        gr.Warning(message)
        return None, message, None

    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    prompt_info = None
    if prompt_wav is not None:
        prompt_wav, prompt_info, prompt_error = prepare_prompt_wav(prompt_wav)
        if prompt_error is not None:
            yield fail(prompt_error)
            return
    if MODEL_IS_COSYVOICE3:
        if mode_checkbox_group in ['3s极速复刻', '跨语种复刻', '自然语言控制']:
            if prompt_wav is None:
                yield fail('prompt音频为空，您是否忘记输入prompt音频？')
                return
            if prompt_info.sample_rate < prompt_sr:
                yield fail('prompt音频采样率{}低于{}'.format(prompt_info.sample_rate, prompt_sr))
                return
            prompt_duration = prompt_info.num_frames / prompt_info.sample_rate
            if prompt_duration < 3:
                yield fail('prompt音频时长{:.2f}s过短，请使用至少 3 秒左右的录音'.format(prompt_duration))
                return
        if mode_checkbox_group == '3s极速复刻':
            if prompt_text == '':
                yield fail('prompt字幕为空，您是否忘记输入prompt字幕？')
                return
            prompt_text = ensure_endofprompt(prompt_text)
        elif mode_checkbox_group == '自然语言控制':
            if instruct_text == '':
                yield fail('您正在使用自然语言控制模式, 请输入instruct文本')
                return
            instruct_text = ensure_endofprompt(instruct_text)
        elif mode_checkbox_group == '跨语种复刻':
            if instruct_text != '':
                gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
            tts_text = ensure_cosyvoice3_cross_lingual_text(tts_text)
        else:
            yield fail('CosyVoice3 不提供预训练音色，请切换到 3s极速复刻 / 跨语种复刻 / 自然语言控制')
            return
    else:
        # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
        if mode_checkbox_group in ['自然语言控制']:
            if instruct_text == '':
                yield fail('您正在使用自然语言控制模式, 请输入instruct文本')
                return
            if prompt_wav is not None or prompt_text != '':
                gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
        # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
        if mode_checkbox_group in ['跨语种复刻']:
            if instruct_text != '':
                gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
            if prompt_wav is None:
                yield fail('您正在使用跨语种复刻模式, 请提供prompt音频')
                return
            gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
        # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
        if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
            if prompt_wav is None:
                yield fail('prompt音频为空，您是否忘记输入prompt音频？')
                return
            if prompt_info.sample_rate < prompt_sr:
                yield fail('prompt音频采样率{}低于{}'.format(prompt_info.sample_rate, prompt_sr))
                return
        # sft mode only use sft_dropdown
        if mode_checkbox_group in ['预训练音色']:
            if instruct_text != '' or prompt_wav is not None or prompt_text != '':
                gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
            if sft_dropdown == '':
                yield fail('没有可用的预训练音色！')
                return
        # zero_shot mode only use prompt_wav prompt text
        if mode_checkbox_group in ['3s极速复刻']:
            if prompt_text == '':
                yield fail('prompt文本为空，您是否忘记输入prompt文本？')
                return
            if instruct_text != '':
                gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    cancel_event = begin_generation()
    if cancel_event is None:
        yield None, '上一个生成任务仍在运行或停止中，请稍等当前段结束后再重新生成', None
        return

    def chunk_iter():
        if not MODEL_IS_COSYVOICE3 and mode_checkbox_group == '预训练音色':
            logging.info('get sft inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                yield i['tts_speech'].numpy().flatten()
        elif mode_checkbox_group == '3s极速复刻':
            logging.info('get zero_shot inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
                yield i['tts_speech'].numpy().flatten()
        elif mode_checkbox_group == '跨语种复刻':
            logging.info('get cross_lingual inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
                yield i['tts_speech'].numpy().flatten()
        elif MODEL_IS_COSYVOICE3 and mode_checkbox_group == '自然语言控制':
            logging.info('get instruct2 inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
                yield i['tts_speech'].numpy().flatten()
        else:
            logging.info('get instruct inference request')
            set_all_random_seed(seed)
            for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
                yield i['tts_speech'].numpy().flatten()

    release_in_main = True
    try:
        total_segments = len(cosyvoice.frontend.text_normalize(tts_text, split=True, text_frontend=True))
        if total_segments == 0:
            total_segments = 1
        start_time = time.time()
        yield None, progress_text(0, total_segments, start_time), None

        if stream is False:
            audio_chunks = []
            completed = 0
            last_audio_path = None
            estimate = None
            chunk_queue = queue.Queue()

            def collect_chunks():
                try:
                    for chunk in chunk_iter():
                        if cancel_event.is_set():
                            chunk_queue.put(('cancelled', None))
                            return
                        chunk_queue.put(('chunk', chunk))
                        if cancel_event.wait(PARTIAL_PREVIEW_PAUSE_SECONDS):
                            chunk_queue.put(('cancelled', None))
                            return
                        if cancel_event.is_set():
                            chunk_queue.put(('cancelled', None))
                            return
                    chunk_queue.put(('done', None))
                except Exception as exc:
                    chunk_queue.put(('error', exc))
                finally:
                    finish_generation(cancel_event)

            release_in_main = False
            worker = threading.Thread(target=collect_chunks, daemon=True)
            worker.start()

            while True:
                try:
                    event, payload = chunk_queue.get(timeout=1.0)
                except queue.Empty:
                    if cancel_event.is_set():
                        yield gr.skip(), cancel_progress_text(completed, total_segments, start_time), None
                        return
                    yield gr.skip(), progress_text(completed, total_segments, start_time, estimate), gr.skip()
                    continue

                if event == 'chunk':
                    if cancel_event.is_set():
                        yield gr.skip(), cancel_progress_text(completed, total_segments, start_time), None
                        return
                    completed += 1
                    audio_chunks.append(payload)
                    estimate = update_estimate(completed, total_segments, start_time)
                    current_progress = progress_text(completed, total_segments, start_time, estimate)
                    logging.info(current_progress)
                    notification = None
                    if completed >= total_segments:
                        gr.Info('音频生成完成')
                        notification = ensure_notification_audio()
                    last_audio_path = save_audio(np.concatenate(audio_chunks), cosyvoice.sample_rate)
                    yield gr.update(value=last_audio_path), current_progress, notification
                    continue

                if event == 'done':
                    break

                if event == 'cancelled':
                    yield gr.skip(), cancel_progress_text(completed, total_segments, start_time), None
                    return

                raise payload

            if completed == 0:
                yield None, progress_text(0, total_segments, start_time), None
            elif last_audio_path is not None:
                yield gr.update(value=last_audio_path), progress_text(completed, total_segments, start_time, estimate), gr.skip()
            return

        completed = 0
        estimate = None
        for chunk in chunk_iter():
            if cancel_event.is_set():
                yield gr.skip(), cancel_progress_text(completed, total_segments, start_time), None
                return
            completed += 1
            estimate = update_estimate(completed, total_segments, start_time)
            current_progress = progress_text(completed, total_segments, start_time, estimate)
            logging.info(current_progress)
            notification = None
            if completed >= total_segments:
                gr.Info('音频生成完成')
                notification = ensure_notification_audio()
            yield gr.update(value=save_audio(chunk, cosyvoice.sample_rate)), current_progress, notification
    except Exception as e:
        logging.exception('generation failed')
        yield None, '生成失败: {}'.format(e), None
    finally:
        if release_in_main:
            finish_generation(cancel_event)


def main():
    with gr.Blocks() as demo:
        if MODEL_IS_COSYVOICE3:
            gr.Markdown("### CosyVoice 3.0 [Modelscope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) \
                        [HuggingFace](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)")
        else:
            gr.Markdown("### 代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                        预训练模型 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                        [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                        [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 请输入需要合成的文本，选择推理模式，并按照提示步骤进行操作")

        tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
        with gr.Row():
            mode_choices = ['3s极速复刻', '跨语种复刻', '自然语言控制'] if MODEL_IS_COSYVOICE3 else inference_mode_list
            default_mode = '3s极速复刻' if MODEL_IS_COSYVOICE3 else inference_mode_list[0]
            mode_checkbox_group = gr.Radio(choices=mode_choices, label='选择推理模式', value=default_mode)
            instruction_text = gr.Text(label="操作步骤", value=instruct_dict[default_mode], scale=0.5)
            if MODEL_IS_COSYVOICE3:
                sft_dropdown = gr.Textbox(value='', visible=False)
            else:
                sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='是否流式推理', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="随机推理种子")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
        prompt_text = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本", value='')
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本", value='')

        with gr.Row():
            generate_button = gr.Button("生成音频")
            stop_button = gr.Button("中断生成", variant="stop")

        audio_output = gr.Audio(label="合成音频", autoplay=False, streaming=False)
        progress_output = gr.Textbox(label="进度", value="等待中", interactive=False)
        notification_output = gr.Audio(label="完成提示音", autoplay=True, visible=False, show_download_button=False)

        seed_button.click(generate_seed, inputs=[], outputs=seed, queue=False)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output, progress_output, notification_output])
        stop_button.click(request_cancel_generation, inputs=[], outputs=[progress_output], queue=False)
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port, allowed_paths=[OUTPUT_DIR])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)
    MODEL_IS_COSYVOICE3 = cosyvoice.__class__.__name__ == 'CosyVoice3'
    MODEL_NAME = cosyvoice.__class__.__name__
    if MODEL_IS_COSYVOICE3:
        instruct_dict.update({
            '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
            '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
            '自然语言控制': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入instruct文本\n3. 点击生成音频按钮',
        })

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()

#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from st_pages import Page, show_pages
from streamlit_extras.bottom_container import bottom
from streamlit_extras.mention import mention

from openlrc import LRCer
from openlrc.gui_streamlit.utils import get_asr_options, get_vad_options, get_preprocess_options, zip_files

show_pages([
    Page("home.py", "Home", "🏠"),
])

with bottom():
    pass

# UI Title
"# openlrc 🎙📄"

'*Transcribe and translate voice into LRC file using Whisper and LLMs (GPT, Claude, et,al).*'

mention(
    label="zh-plus/openlrc",
    icon="github",  # GitHub is also featured!
    url="https://github.com/zh-plus/openlrc",
)

# Sidebar - Normal Configuration
st.sidebar.header('Configuration')

with st.sidebar.popover("API Keys"):
    openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get('OPENAI_API_KEY'), key='openai_api')
    anthropic_api_key = st.text_input("Anthropic API Key", value=os.environ.get('ANTHROPIC_API_KEY'),
                                      key='anthropic_api')

whisper_model = st.sidebar.selectbox('Whisper Model',
                                     ['large-v3', 'medium', 'medium.en', 'small', 'small.en', 'base', 'base.en', 'tiny',
                                      'tiny.en'], index=0, key='whisper_model')
compute_type = st.sidebar.selectbox('Compute Type', ['int8', 'int8_float16', 'int16', 'float16', 'float32'], index=3
                                    , key='compute_type')

if not openai_api_key and not anthropic_api_key:
    chatbot_model = st.sidebar.selectbox('Chatbot Model', [], disabled=True, help="Please provide an API key first.",
                                         key='chatbot_model')
elif openai_api_key and not anthropic_api_key:
    chatbot_model = st.sidebar.selectbox('Chatbot Model',
                                         ['gpt-3.5-turbo', 'gpt-4-0125-preview', 'gpt-4-turbo-preview'], index=0,
                                         help="Model for translation. Check [pricing](/pricing) for more details.",
                                         key='chatbot_model')
elif not openai_api_key and anthropic_api_key:
    chatbot_model = st.sidebar.selectbox('Chatbot Model', ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229',
                                                           'claude-3-opus-20240229'], index=0,
                                         help="Model for translation. Check [pricing](/pricing) for more details.",
                                         key='chatbot_model')
else:
    chatbot_model = st.sidebar.selectbox('Chatbot Model',
                                         ['gpt-3.5-turbo', 'gpt-4-0125-preview', 'gpt-4-turbo-preview',
                                          'claude-3-haiku-20240307', 'claude-3-sonnet-20240229',
                                          'claude-3-opus-20240229'],
                                         index=0,
                                         help="Model for translation. Check [pricing](/pricing) for more details.",
                                         key='chatbot_model')

# fee_limit = st.sidebar.number_input('Fee Limit', min_value=0.0, value=0.1, step=0.01)
fee_limit = st.sidebar.slider('Fee Limit (USD)', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='fee_limit')
consumer_thread = st.sidebar.slider('Consumer Thread', min_value=1, max_value=12, value=4, step=1,
                                    key='consumer_thread')
proxy = st.sidebar.text_input('Proxy', help='e.g.: http://127.0.0.1:7890', key='proxy')

# Sidebar: Advanced Configuration
with st.sidebar.expander("Advanced Configuration", expanded=False):
    st.write("### ASR Options")
    # ASR Options with help messages
    beam_size = st.number_input("Beam Size", value=3, min_value=1, help="Sets the beam size for the ASR.")
    best_of = st.number_input("Best of", value=5, min_value=1, help="Chooses the best result from N beams.")
    patience = st.number_input("Patience", value=1, min_value=0,
                               help="Number of times to retry ASR for better accuracy.")
    length_penalty = st.number_input("Length Penalty", value=1, min_value=0,
                                     help="Penalty for longer sequences to control output length.")
    repetition_penalty = st.number_input("Repetition Penalty", value=1.0, min_value=0.0,
                                         help="Penalty for repeated tokens to encourage diversity.")
    no_repeat_ngram_size = st.number_input("No Repeat Ngram Size", value=0, min_value=0,
                                           help="Size of ngram that cannot be repeated in the output.")
    temperature = st.number_input("Temperature", value=0.0, min_value=0.0, max_value=1.0, step=0.01,
                                  help="Controls randomness in beam search. Lower values make results more deterministic.")
    compression_ratio_threshold = st.number_input("Compression Ratio Threshold", value=None, min_value=0.0,
                                                  help="Threshold for compression ratio.")
    log_prob_threshold = st.number_input("Log Prob Threshold", value=None, min_value=0.0,
                                         help="Threshold for log probability.")
    no_speech_threshold = st.number_input("No Speech Threshold", value=None, min_value=0.0,
                                          help="Threshold for no speech.")
    condition_on_previous_text = st.checkbox("Condition on Previous Text", value=False,
                                             help="The previous output of the model will be provided as a prompt for the next window. (May enhance model hallucinations)")
    initial_prompt = st.text_input("Initial Prompt", value=None, help="Initial prompt for the ASR.")
    prefix = st.text_input("Prefix", value=None, help="Prefix for the ASR.")
    suppress_blank = st.checkbox("Suppress Blank", value=True, help="Whether to suppress blank tokens in the output.")
    suppress_tokens = st.text_input("Suppress Tokens", value="-1",
                                    help="Tokens to be suppressed in the output, int values split by ','.")
    without_timestamps = st.checkbox("Without Timestamps", value=False, help="Generate output without timestamps.")
    max_initial_timestamp = st.number_input("Max Initial Timestamp", min_value=0.0, value=0.0, step=0.01,
                                            help="Maximum initial timestamp for the output.")
    word_timestamps = st.checkbox("Word Timestamps", value=True,
                                  help="Whether to include timestamps for each word in the output.")
    prepend_punctuations = st.text_input("Prepend Punctuations", value="\"'“¿([{-",
                                         help="Punctuations to prepend to the output.")
    append_punctuations = st.text_input("Append Punctuations", value="\"'.。,，!！?？:：”)]}、",
                                        help="Punctuations to append to the output.")
    hallucination_silence_threshold = st.number_input("Hallucination Silence Threshold", min_value=0, value=2,
                                                      help="Threshold for determining hallucinations in silence periods.")

    # VAD Options with help messages
    st.write("### VAD Options")
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.382,
                          help="Threshold for voice activity detection.")
    min_speech_duration_ms = st.number_input("Min Speech Duration (ms)", min_value=0, value=250,
                                             help="Minimum duration of speech for it to be considered valid.")
    max_speech_duration_s = st.number_input("Max Speech Duration (s)", min_value=0.0, value=1e+308, format="%e",
                                            help="Maximum duration of speech. Use a high value to indicate no limit.")
    min_silence_duration_ms = st.number_input("Min Silence Duration (ms)", min_value=0, value=2000,
                                              help="Minimum duration of silence for splitting phrases.")
    window_size_samples = st.number_input("Window Size Samples", min_value=1, value=1024,
                                          help="Size of the analysis window for VAD.")
    speech_pad_ms = st.number_input("Speech Pad (ms)", min_value=0, value=400, help="Padding added to speech segments.")

    # Preprocess Options with help messages
    st.write("### Preprocess Options")
    atten_lim_db = st.number_input("Atten Lim DB", value=15, min_value=0, help="Limit for attenuation in decibels.")

with st.form("transcribe_translate_form"):
    st.write("## Transcribe and Translate")

    # File upload (assuming 'paths' will be handled separately or is part of the form)
    files = st.file_uploader("Upload files", accept_multiple_files=True,
                             type=['mp3', 'wav', 'flac', 'm4a', 'mp4', 'avi', 'mkv', 'webm', 'mov', 'wmv', 'flv', ])

    # Currently, st.file_uploader cant return paths. (On planning in https://roadmap.streamlit.app/#future)
    # Thus, save them into temporary files and use lrcer.run on them
    tmpdir = tempfile.mkdtemp()  # Remember to delete it when the process is done
    paths = []
    for file in files:
        with open(os.path.join(tmpdir, file.name), 'wb') as f:
            f.write(file.read())
        paths.append(os.path.join(tmpdir, file.name))

    src_lang = st.selectbox("Source Language",
                            options=['Auto Detect', 'ca', 'zh', 'hr', 'da', 'nl', 'en', 'fi', 'fr', 'de', 'el', 'it',
                                     'ja', 'ko', 'lt', 'mk', 'nb', 'pl', 'pt', 'ro', 'ru', 'sl', 'es', 'sv', 'uk'],
                            index=0,
                            format_func=lambda x: 'Auto Detect' if x == 'Auto Detect' else x.upper(),
                            help='Currently bottleneck-ed by Spacy')
    target_lang = st.text_input("Target Language", value='zh-cn', help='Language code for translation target')
    prompter = st.selectbox("Prompter", options=['base'], disabled=True, help='Currently, only `base` is supported.')

    col1, col2, col3 = st.columns(3)
    with col1:
        skip_trans = st.checkbox("Skip Translation")
    with col2:
        noise_suppress = st.checkbox("Noise Suppression")
    with col3:
        bilingual_sub = st.checkbox("Bilingual Subtitles")

    # Form submission button
    submitted = st.form_submit_button("GO!", type='primary')

if submitted:
    # Assuming 'paths' is defined or obtained elsewhere in your app
    src_lang = None if src_lang == 'Auto Detect' else src_lang

    with st.spinner('Running...'):
        lrcer = LRCer(whisper_model=whisper_model, compute_type=compute_type, chatbot_model=chatbot_model,
                      fee_limit=fee_limit, consumer_thread=consumer_thread,
                      asr_options=get_asr_options(
                          beam_size, best_of, patience, length_penalty, repetition_penalty, no_repeat_ngram_size,
                          temperature, compression_ratio_threshold, log_prob_threshold, no_speech_threshold,
                          condition_on_previous_text, initial_prompt, prefix, suppress_blank, suppress_tokens,
                          without_timestamps, max_initial_timestamp, word_timestamps, prepend_punctuations,
                          append_punctuations, hallucination_silence_threshold),
                      vad_options=get_vad_options(
                          threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms,
                          window_size_samples, speech_pad_ms),
                      preprocess_options=get_preprocess_options(atten_lim_db),
                      proxy=proxy, )
        results = lrcer.run(paths, src_lang=src_lang, target_lang=target_lang, prompter=prompter,
                            skip_trans=skip_trans, noise_suppress=noise_suppress, bilingual_sub=bilingual_sub)
    print(paths)
    print(results)

    result_file_path = results[0] if len(results) == 1 else zip_files(results)

    print(result_file_path)

    with open(result_file_path, 'rb') as f:
        st.download_button("Download LRC Files", f, file_name=Path(result_file_path).name)

    # Remove tmpdir contents
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f'Removed {tmpdir}.')

from openlrc import LRCer

if __name__ == '__main__':
    lrcer = LRCer(chatbot_model='gpt-4o',
                  base_url_config={'openai': 'https://api.gptsapi.net/v1'},
                #   device='cpu',
                #   compute_type='float32',
                  #is_force_glossary_used=True
                  )

    lrcer.run(['./data/test.mp3'], target_lang='zh-cn', skip_trans=False,video_understanding = True, sampling_frequency=5)
    # Generate translated ./data/test_audio.lrc and ./data/test_video.srt

    # Use glossary to improve translation
    # lrcer = LRCer(glossary='/data/aoe4-glossary.yaml')
    #
    # # To skip translation process
    # #lrcer.run('./data/test.mp3', target_lang='en', skip_trans=True)
    #
    # # Change asr_options or vad_options, check openlrc.defaults for details
    # vad_options = {"threshold": 0.1}
    # lrcer = LRCer(vad_options=vad_options)
    # lrcer.run('./data/test.mp3', target_lang='zh-cn')
    #
    # # Enhance the audio using noise suppression (consume more time).
    # lrcer.run('./data/test.mp3', target_lang='zh-cn', noise_suppress=True)
    #
    # # Change the LLM model for translation
    # lrcer = LRCer(chatbot_model='claude-3-sonnet-20240229')
    # lrcer.run('./data/test.mp3', target_lang='zh-cn')
    #
    # # Clear temp folder after processing done
    # lrcer.run('./data/test.mp3', target_lang='zh-cn', clear_temp=True)
    #
    # # Change base_url
    # lrcer = LRCer(base_url_config={'openai': 'https://api.g4f.icu/v1',
    #                                'anthropic': 'https://example/api'})
    #
    # # Route model to arbitrary Chatbot SDK
    # lrcer = LRCer(chatbot_model='openai: claude-3-sonnet-20240229',
    #               base_url_config={'openai': 'https://api.g4f.icu/v1/'})
    #
    # # Bilingual subtitle
    # lrcer.run('./data/test.mp3', target_lang='zh-cn', bilingual_sub=True)
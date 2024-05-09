#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import streamlit as st

st.set_page_config(
    page_title="Pricing xd",
    page_icon="💰",
    layout="wide",
)

chatbot_help_msg = """
## Pricing 💰

*Pricing data from [OpenAI](https://openai.com/pricing) and [Anthropic](https://docs.anthropic.com/claude/docs/models-overview#model-comparison)*

| Model Name                 | Pricing for 1M Tokens (Input/Output) (USD) | Cost for 1 Hour Audio (USD) |
|----------------------------|-------------------------------------------------|----------------------------------|
| `gpt-3.5-turbo-0125`       | 0.5, 1.5                                        | 0.01                             |
| `gpt-3.5-turbo`            | 0.5, 1.5                                        | 0.01                             |
| `gpt-4-0125-preview`       | 10, 30                                          | 0.5                              |
| `gpt-4-turbo-preview`      | 10, 30                                          | 0.5                              |
| `claude-3-haiku-20240307`  | 0.25, 1.25                                      | 0.015                            |
| `claude-3-sonnet-20240229` | 3, 15                                           | 0.2                              |
| `claude-3-opus-20240229`   | 15, 75                                          | 1                                |

**Note: The cost is estimated based on the token count of the input and output text. The actual cost may vary due to the language and audio speed.**

### Recommended Translation Model

- For English audio, we recommend using `gpt-3.5-turbo`.
- For non-English audio, we recommend using `claude-3-sonnet-20240229`.
"""
st.markdown(chatbot_help_msg, unsafe_allow_html=True)

#  Copyright (C) 2024. Hao Zheng
#  All rights reserved.

import streamlit as st

context_msg = """## Context 📝

Utilize the available context to enhance the quality of your translation.
Save them as `context.yaml` in the same directory as your audio file.

> [!NOTE]
> The improvement of translation quality from Context is **NOT** guaranteed.

```yaml
background: "This is a multi-line background.
This is a basic example."
audio_type: Movie
description_map: {
  movie_name1 (without extension): "This
  is a multi-line description for movie1.",
  movie_name2 (without extension): "This
  is a multi-line description for movie2.",
  movie_name3 (without extension): "This is a single-line description for movie 3.",
}
```"""
st.write(context_msg)

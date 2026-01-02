BG_COLOR = "#050505"
USER_BG_COLOR = "#121212"
BORDER_COLOR = "#262626"
ACCENT_COLOR = "#78beb4"
TEXT_COLOR = "#a8a29e"

CSS = f"""
/* Global Screen Style */
Screen {{
    background: {BG_COLOR};
    layout: vertical;
}}

/* --- Model Selection Screen --- */
ModelSelectScreen {{
    align: center middle;
}}

#model-select-container {{
    width: 60;
    height: auto;
    max-height: 90%;
    border: solid {BORDER_COLOR};
    background: {BG_COLOR};
    padding: 1 2;
    layout: vertical;
    overflow-y: auto;
    scrollbar-size: 0 0;
}}

.title-label {{
    color: {ACCENT_COLOR};
    text-style: bold;
    text-align: center;
    margin-bottom: 2;
    width: 100%;
}}

.section-label {{
    color: {TEXT_COLOR};
    margin-top: 1;
    margin-bottom: 1;
    opacity: 0.6; /* Lower opacity for 'darker/off' look */
    width: 100%;
    text-align: center;
}}

/* Styled Buttons */
Button {{
    width: 100%;
    margin-bottom: 1;
    background: {USER_BG_COLOR};
    color: {TEXT_COLOR};
    border: none;
    text-align: center;
}}

Button:hover {{
    background: {ACCENT_COLOR};
    color: {BG_COLOR};
    text-style: bold;
}}

/* Input Field Style */
#model-input {{
    width: 100%;
    margin-top: 1;
    border: solid {BORDER_COLOR};
    background: transparent;
    color: {TEXT_COLOR};
    text-align: center;
}}

#model-input:focus {{
    border: solid {ACCENT_COLOR};
}}

/* --- Chat Screen Styles --- */
#chat-container {{
    height: 1fr;
    width: 100%;
    scrollbar-size: 1 1;
    scrollbar-color: {ACCENT_COLOR}; 
}}

.user-msg {{
    background: {USER_BG_COLOR};
    color: {TEXT_COLOR};
    padding: 1 2;
    margin-top: 1;
    width: 100%;
    border-left: solid {ACCENT_COLOR};
}}

.ai-msg {{
    background: {BG_COLOR};
    color: {TEXT_COLOR};
    padding: 0 2;
    margin-top: 1;
    width: 100%;
}}

.ai-label {{
    color: {ACCENT_COLOR};
    text-style: bold;
    margin-bottom: 0;
}}

#input-area {{
    dock: bottom;
    height: 3;
    width: 100%;
    background: {BG_COLOR};
    border-top: solid {BORDER_COLOR};
    padding: 0 1;
    align-vertical: middle;
}}

#prompt-label {{
    color: {ACCENT_COLOR};
    text-style: bold;
    width: 2;
    padding-top: 1; 
}}

#chat-input {{
    width: 1fr;
    background: transparent;
    border: none;
    color: {TEXT_COLOR};
    height: 1;
    padding: 0;
}}
"""
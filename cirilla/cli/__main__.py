import re
import json
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.containers import VerticalScroll, Horizontal, Container
from textual.widgets import Input, Static, Markdown, Label, Button
from .theme import CSS
from cirilla.Cirilla_model import Cirilla
from cirilla.Cirilla_model import CirillaTokenizer
from cirilla.Cirilla_model.modules import select_torch_device

SAVED_HISTORY = ["Cirilla"]

def save_to_history_in_file(model_name: str):
    global SAVED_HISTORY
    
    current = list(SAVED_HISTORY)
    if model_name in current:
        current.remove(model_name)
    current.insert(0, model_name)
    current = current[:3]
    SAVED_HISTORY = current

    try:
        with open(__file__, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = r"^SAVED_HISTORY = \[.*?\]"
        replacement = f"SAVED_HISTORY = {json.dumps(current)}"
        
        new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)

        with open(__file__, "w", encoding="utf-8") as f:
            f.write(new_content)
            
    except Exception:
        pass

class ModelSelectScreen(Screen):
    
    def compose(self) -> ComposeResult:
        with Container(id="model-select-container"):
            yield Label("Cirilla Vibe", classes="title-label")
            
            yield Label("Featured Models:", classes="section-label")
            yield Button("Cirilla-0.3B-4E", id="btn-large")

            if SAVED_HISTORY:
                yield Label("Recent Inputs:", classes="section-label")
                for model in SAVED_HISTORY:
                    yield Button(model, classes="history-btn")

            yield Label("Or type custom model name:", classes="section-label")
            yield Input(placeholder="Model Name", id="model-input")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        model_name = str(event.button.label)
        self.app.push_screen(ChatScreen(model_name))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        model_name = event.value.strip()
        if model_name:
            save_to_history_in_file(model_name)
            self.app.push_screen(ChatScreen(model_name))

class ChatScreen(Screen):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = Cirilla()
        self.hub_url = f'AnthonyPa57/{model_name}' if '/' not in model_name else model_name
        self.model.pull_model_from_hub(self.hub_url, inference_mode=True, map_device=select_torch_device())
        self.tokenizer = CirillaTokenizer(hub_url=self.hub_url)
        self.termination_tokens = [self.tokenizer.convert_tokens_to_ids('<eos>'), self.tokenizer.convert_tokens_to_ids('<|user|>')]
        self.history = []
        self.generation_config = {
                                'kv_cache':True,
                                'top_p':0.05,
                                'top_k':None,
                                'temperature':0.7,
                                'n_beams':9,
                                'auto_clear':True
                                }
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-container"):
            yield Static(self.model_name, classes="ai-label ai-msg")
            yield Markdown(f"Connected to **{self.model_name}**.\nStart typing to interact.", classes="ai-msg")

        with Horizontal(id="input-area"):
            yield Label(">", id="prompt-label")
            yield Input(placeholder="Type a message...", id="chat-input")

    def on_mount(self):
        self.query_one("#chat-input").focus()

    async def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text:
            return
        
        if text.startswith("/set"):
            try:
                parts = text.split(" ")
                if len(parts) < 2 or "=" not in parts[1]:
                    await self.app.bell()
                    return

                command_part = parts[1]
                key, val = command_part.split("=")

                if key == 'kv_cache':
                    self.generation_config['kv_cache'] = val.lower() == "true"
                elif key == 'top_p':
                    self.generation_config['top_p'] = float(val)
                elif key == 'top_k':
                    self.generation_config['top_k'] = int(val)
                elif key == 'temperature':
                    self.generation_config['temperature'] = float(val)
                elif key == 'n_beams':
                    self.generation_config['n_beams'] = int(val)
                elif key == 'auto_clear':
                    self.generation_config['auto_clear'] = val.lower() == "true"
                else:
                    await self.app.bell()
                    return
                
                self.notify(f"Set {key} to {val}")
                event.input.value = "" 
                return
            
            except Exception:
                await self.app.bell()
                return
            
        elif text.startswith("/clear"):
            self.history = []
            self.model.clear_cache()
            event.input.value = ""
            self.notify("Chat history cleared.")
            return
        
        else:

            chat = self.query_one("#chat-container")
            await chat.mount(Markdown(text, classes="user-msg"))

            if self.generation_config['auto_clear']:
                self.history = []
                self.model.clear_cache()

            self.history.append({
                "role": "user",
                "content": text})

            event.input.value = ""

            if self.generation_config['kv_cache']:
                batch_prompts = [self.history for _ in range(self.generation_config['n_beams'])]
                x = self.tokenizer.apply_chat_template(batch_prompts, padding='do_not_pad', add_generation_prompt=True)
                out = self.model.generate_kv_cache(x, termination_tokens=self.termination_tokens,
                                                        top_k=self.generation_config['top_k'],
                                                        top_p=self.generation_config['top_p'],
                                                        temperature=self.generation_config['temperature'],
                                                        sample_parallel=self.generation_config['n_beams'] > 1)
                if self.generation_config['n_beams'] == 1:
                    out = out[0]

                input_length = len(x[0])

            else:
                x = self.tokenizer.apply_chat_template(self.history, padding='do_not_pad', add_generation_prompt=True, return_tensors='pt')
                out = self.model.generate_naive(x.to(self.model.args.device),
                                                termination_tokens=self.termination_tokens,
                                                top_k=self.generation_config['top_k'],
                                                top_p=self.generation_config['top_p'],
                                                n_beams=self.generation_config['n_beams'],
                                                temperature=self.generation_config['temperature'])[0]
                input_length = x.shape[1]

            text = self.tokenizer.decode(out[input_length:])\
                                            .replace('<pad>', '')\
                                            .replace('<|user|>', '')\
                                            .replace('<|assistant|>', '')\
                                            .replace('<eos>', '')\
                                            .replace('<sos>', '')\
                                            .replace('<unk>', '')\
                                            .strip()
            
            self.history.append({
                "role": "assistant",
                "content": text})

            await chat.mount(Static(self.model_name, classes="ai-label ai-msg"))
            await chat.mount(Markdown(text, classes="ai-msg"))

            chat.scroll_end(animate=True)

class SimpleVibeApp(App):
    CSS = CSS

    def on_mount(self):
        self.push_screen(ModelSelectScreen())

if __name__ == "__main__":
    app = SimpleVibeApp()
    app.run()
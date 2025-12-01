import gradio as gr
import torch
import os
import datetime
import yaml
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SmolVLM2")

class App:
    def __init__(self, config_path: str = "scripts/config.yaml"):

        self.cfg = self._load_config(config_path)
        self._setup_device()
        self.cache_dir = os.getenv("HF_HOME")
        
        logger.info("App startup")
        logger.info(f"Use Device: {self.device} Dtype: {self.dtype}")
        if self.cache_dir:
            logger.info(f"Cache Dir: {self.cache_dir}")
            
        self._load_model()

    def _load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            logger.error(f"Configuration file not found at {path}!")
            exit(1)
        
        with open(path, "r", encoding = "utf-8") as f:
            return yaml.safe_load(f)

    def _setup_device(self) -> None:
        if self.cfg["model"]["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.cfg["model"]["device"]

        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def _load_model(self) -> None:
        try:
            model_id = self.cfg["model"]["id"]
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                _attn_implementation="sdpa",
                cache_dir=self.cache_dir
            ).to(self.device)
        except Exception as e:
            logger.critical(f"Error while loading model: {e}")
            exit(1)

    def process_video_chat(self, video_path: str, user_prompt: str) -> tuple:
        ui_logs = []
        
        def log(msg, level = logging.INFO):
            # helper function for logging in gradio web interface
            logger.log(level, msg)
            time_str = datetime.datetime.now().strftime("%H:%M:%S")
            lvl_name = logging.getLevelName(level)
            ui_logs.append(f"[{time_str}] [{lvl_name}] {msg}")

        if not video_path:
            return "Upload video!", "\n".join(ui_logs)
        
        final_prompt = user_prompt if user_prompt else "Describe this video."
        log(f"Prompt: {final_prompt}")

        try:
            messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "video", "path": video_path}, 
                        {"type": "text", "text": final_prompt}
                    ]
                }]
            
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.device, dtype=self.dtype)
            
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=256
                )
            
            output_ids = generated_ids[:, input_len:]
            answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            log("Done.")
            return answer, "\n".join(ui_logs)

        except Exception as e:
            log(f"Error: {e}", logging.ERROR)
            return "Error.", "\n".join(ui_logs)

    def process_image_ocr(self, image_path: str) -> tuple:
        ui_logs = []
        
        def log(msg, level = logging.INFO):
            # helper function for logging in gradio web interface
            logger.log(level, msg)
            time_str = datetime.datetime.now().strftime("%H:%M:%S")
            lvl_name = logging.getLevelName(level)
            ui_logs.append(f"[{time_str}] [{lvl_name}] {msg}")

        system_prompt = "Transcribe all text from this image exactly as it appears on the image. Do not add any other text."

        if not image_path:
            return None, "\n".join(ui_logs)

        log(f"OCR started.")

        try:
            image = Image.open(image_path)
            
            messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image}, 
                        {"type": "text", "text": system_prompt}
                    ]
            }]

            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.device, dtype=self.dtype)
            
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    do_sample=False, 
                    max_new_tokens=512
                )

            output_ids = generated_ids[:, input_len:]
            result_text = self.processor.decode(output_ids[0], skip_special_tokens=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ocr_result_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result_text)
                
            log(f"Result saved to {filename}")
            return filename, "\n".join(ui_logs)

        except Exception as e:
            log(f"Error: {str(e)}", logging.ERROR)
            return None, "\n".join(ui_logs)

    def launch(self):
        with gr.Blocks(title="SmolVLM2-Demo") as demo:
            gr.Markdown("## SmolVLM2-Demo")
            
            with gr.Tabs():
                with gr.TabItem("Video-chat"):
                    with gr.Row():
                        with gr.Column():
                            v_input = gr.Video(label="Video")
                            v_text = gr.Textbox(
                                label="Prompt", 
                                value="Describe this video."
                            )
                            v_btn = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            v_out = gr.Textbox(label="Output", lines=10)
                            v_log = gr.Textbox(label="Logs", lines=5, elem_id="log_box")
                    
                    v_btn.click(self.process_video_chat, [v_input, v_text], [v_out, v_log])

                with gr.TabItem("OCR"):
                    with gr.Row():
                        with gr.Column():
                            i_input = gr.Image(type="filepath", label="Image")
                            i_btn = gr.Button("Extract Text to File", variant="primary")
                        with gr.Column():
                            i_out = gr.File(label="Download result (.txt)")
                            i_log = gr.Textbox(label="Logs", lines=5, elem_id="log_box")

                    i_btn.click(self.process_image_ocr, [i_input], [i_out, i_log])

        env_port = os.getenv("PORT") # load from env var
        final_port = int(env_port) if env_port else self.cfg["server"]["port"]
        
        demo.launch(
            server_name=self.cfg["server"]["host"],
            server_port=final_port
        )

if __name__ == "__main__":
    app = App()
    app.launch()
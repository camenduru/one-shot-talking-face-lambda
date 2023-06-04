import os

os.system(f"pip install -q torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U")
os.system(f"git lfs install")
os.chdir(f"/home/demo/source")
os.system(f"git clone https://huggingface.co/camenduru/pocketsphinx-20.04-a10 pocketsphinx")
os.chdir(f"/home/demo/source/pocketsphinx")
os.system(f"sudo cmake --build build --target install")
os.chdir(f"/home/demo/source")
os.system(f"git clone https://huggingface.co/camenduru/one-shot-talking-face-20.04-a10 one-shot-talking-face")
os.chdir(f"/home/demo/source/one-shot-talking-face")
os.system(f"pip install -r /home/demo/source/one-shot-talking-face/requirements.txt")
os.system(f"chmod 755 /home/demo/source/one-shot-talking-face/OpenFace/FeatureExtraction")
os.system(f"mkdir /home/demo/source/train")
os.system(f"pip install -q imageio-ffmpeg gradio numpy==1.23.0")

os.system(f"wget https://github.com/camenduru/one-shot-talking-face-colab/raw/main/test/audio.wav -O /home/demo/source/audio.wav")
os.system(f"wget https://github.com/camenduru/one-shot-talking-face-colab/blob/main/test/image.png?raw=true -O /home/demo/source/image.png")


import gradio as gr
import os, subprocess, torchaudio
import torch
from PIL import Image

block = gr.Blocks()

def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image

def calculate(image_in, audio_in):
    waveform, sample_rate = torchaudio.load(audio_in)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    torchaudio.save("/content/audio.wav", waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    image = Image.open(image_in)
    image = pad_image(image)
    image.save("image.png")

    pocketsphinx_run = subprocess.run(['pocketsphinx', '-phone_align', 'yes', 'single', '/content/audio.wav'], check=True, capture_output=True)
    jq_run = subprocess.run(['jq', '[.w[]|{word: (.t | ascii_upcase | sub("<S>"; "sil") | sub("<SIL>"; "sil") | sub("\\\(2\\\)"; "") | sub("\\\(3\\\)"; "") | sub("\\\(4\\\)"; "") | sub("\\\[SPEECH\\\]"; "SIL") | sub("\\\[NOISE\\\]"; "SIL")), phones: [.w[]|{ph: .t | sub("\\\+SPN\\\+"; "SIL") | sub("\\\+NSN\\\+"; "SIL"), bg: (.b*100)|floor, ed: (.b*100+.d*100)|floor}]}]'], input=pocketsphinx_run.stdout, capture_output=True)
    with open("test.json", "w") as f:
        f.write(jq_run.stdout.decode('utf-8').strip())

    os.system(f"cd /content/one-shot-talking-face && python3 -B test_script.py --img_path /content/image.png --audio_path /content/audio.wav --phoneme_path /content/test.json --save_dir /content/train")
    return "/content/train/image_audio.mp4"
    
def run():
  with block:
    gr.Markdown(
    """
    <style> body { text-align: right} </style>
    map: 📄 [arxiv](https://arxiv.org/abs/2112.02749) &nbsp; ⇨ 👩‍💻 [github](https://github.com/FuxiVirtualHuman/AAAI22-one-shot-talking-face) &nbsp; ⇨ 🦒 [colab](https://github.com/camenduru/one-shot-talking-face-colab) &nbsp; ⇨ 🐝 [lambdalabs](https://cloud.lambdalabs.com/demos/camenduru/one-shot-talking-face) &nbsp; | 🐣 [twitter](https://twitter.com/camenduru) &nbsp;
    """)
    with gr.Group():
      with gr.Box():
        with gr.Row().style(equal_height=True):
          image_in = gr.Image(show_label=False, type="filepath")
          audio_in = gr.Audio(show_label=False, type='filepath')
          video_out = gr.Video(show_label=False)
        with gr.Row().style(equal_height=True):
          btn = gr.Button("Generate")          

    examples = gr.Examples(examples=[
      ["./image.png", "./audio.wav"],
    ], fn=calculate, inputs=[image_in, audio_in], outputs=[video_out], cache_examples=True)

    btn.click(calculate, inputs=[image_in, audio_in], outputs=[video_out])
    block.queue()
    block.launch()

if __name__ == "__main__":
    run()
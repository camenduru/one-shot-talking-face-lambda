import gradio as gr

block = gr.Blocks()

def run():
  with block:
    gr.Markdown(
    """
    <p>oh no 😐 something wrong with the 🤗 hugging face servers 😐 hopefully, it will be fixed soon</p>
    """)
    block.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    run()
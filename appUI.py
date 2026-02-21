import gradio as gr

def create_ui(process_fn):
    with gr.Blocks(title="ABCDE Melanoma Analyzer") as demo:
        gr.Markdown("# 🔬 Melanoma ABCDE Diagnostic Assistant")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Dermoscopy Image")
                input_d = gr.Radio(["6mm未満", "6mm以上"], label="D: Diameter", value="6mm未満")
                input_e = gr.Checkbox(label="E: Evolving", value=False)
                btn = gr.Button("解析と診断を実行", variant="primary")
                
            with gr.Column():
                result_img = gr.Image(label="Segmentation Result")
                with gr.Row():
                    score_a = gr.Number(label="A: Asymmetry", value=0.0)
                    score_b = gr.Number(label="B: Border", value=0.0)
                    score_c = gr.Number(label="C: Color", value=0.0)
                final_judgment = gr.Textbox(label="判定結果", interactive=False)

        btn.click(
            process_fn, 
            inputs=[input_img, input_d, input_e], 
            outputs=[result_img, score_a, score_b, score_c, final_judgment]
        )
    return demo
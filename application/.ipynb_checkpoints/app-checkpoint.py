import gradio as gr
import query_to_similar_attack_cwe

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale = 1):
            query_input = gr.Textbox(lines=2, placeholder="Enter your query here...", label="User Query")
            model_dropdown = gr.Dropdown(choices=['GPT-2','SecBERT','SecureBERT'], label="Choose a Text Embedding Model")
            submit_btn = gr.Button("Submit")

        with gr.Column(scale = 1):
            prompt_output = gr.Textbox(label="Related ATT&CK and CWE Descriptions")
            # augmented_output = gr.Textbox(label="OpenAI Response using Augmented Prompt")

        def get_model_response(query, model):
            augmented_prompt = query_to_similar_attack_cwe.main(query, model)

            return {prompt_output: augmented_prompt}

        submit_btn.click(
            get_model_response,
            [query_input, model_dropdown],
            [prompt_output],
        )
demo.launch(server_name="0.0.0.0", server_port=None, share=True)

# prompt = "Can you suggest common weaknesses and vulnerabilities related to the Colonial Pipeline Attack? In May of 2021, a hacker group known as DarkSide gained access to Colonial Pipeline’s network through a compromised VPN password. This was possible, in part, because the system did not have multifactor authentication protocols in place. This made entry into the VPN easier since multiple steps were not required to verify the user’s identity. Even though the compromised password was a “complex password,” malicious actors acquired it as part of a separate data breach."
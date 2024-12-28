import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import sys
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('../')
import config
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        self.activation = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout_rate-0.3)
        self.dropout2 = nn.Dropout(dropout_rate-0.2)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, apply_dropout=False):
        x = self.fc1(x)
        x = self.activation(x)
        if apply_dropout:
            x = self.dropout1(x)

        x = self.fc2(x)
        x = self.activation(x)
        if apply_dropout:
            x = self.dropout2(x)

        x = self.fc3(x)
        x = self.activation(x)
        if apply_dropout:
            x = self.dropout3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dims, output_dim, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], output_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate-0.2)
        self.dropout3 = nn.Dropout(dropout_rate-0.3)

    def forward(self, x, apply_dropout=False):
        if apply_dropout:
            x = self.dropout1(x)
        x = self.fc1(x)

        if apply_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)

        if apply_dropout:
            x = self.dropout3(x)
        x = self.fc3(x)

        return x

def generate_graph_emb(org_emb, encoder, decoder, encoder_model_path):
    checkpoint = torch.load(encoder_model_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.to(device)  # Ensure model is on the correct device
    decoder.to(device)  # Ensure model is on the correct device
    encoder.eval()
    decoder.eval()
    
    new_text_embeddings = torch.tensor(org_emb, dtype=torch.float32).to(device)
    with torch.no_grad():
        encoded_text = encoder(new_text_embeddings.to(device))
        generated_graph_embeddings = decoder(encoded_text)
    return generated_graph_embeddings.cpu().detach().numpy()  # Move tensor to CPU before converting to numpy

def retrieve_embeddings(input_text, model_name):
    if(model_name=="SBERT"):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(input_text)
        return torch.tensor(embeddings)
    else:
        if model_name == "SecureBERT":
            print("Load pretrained_SecureBert model:")
            tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
            model = RobertaForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")
        elif model_name == "SecBERT":
            print("Load pretrained_SecBert model:")
            tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecRoBERTa")
            model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecRoBERTa")
        elif model_name == "GPT-2":
            print("Load pretrained gpt2 model:")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        # gpt2_model_name = "gpt2-xl"
        # model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
        # tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("device: ", device)
        model.to(device)
        model.eval()
        chunk_size=110
        tokens = tokenizer.tokenize(input_text)
        chunk_embeddings = []
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            input_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
            input_tensors = torch.tensor([input_ids]).to(device)
            with torch.no_grad():
                outputs = model(input_tensors, output_hidden_states=True)
            last_layer_embeddings = outputs.hidden_states[-1]
            mean_pooled = last_layer_embeddings.mean(dim=1)
            chunk_embeddings.append(mean_pooled)
        all_embeddings = torch.cat(chunk_embeddings, dim=0)
        prompt_embeddings = all_embeddings.mean(dim=0).squeeze()
        return prompt_embeddings.cpu().detach().numpy()


def get_prompt_related_embeddings(prompt_text_embeddings, text_embeddings, graph_embeddings):
    attack_size = 203
    weakness_size = 933

    # Ensure prompt_text_embeddings is a CPU tensor if it's used with NumPy
    # prompt_text_embeddings = prompt_text_embeddings.cpu().reshape(1, -1)  # Move to CPU and reshape
    print(prompt_text_embeddings.shape)
    #graph_embeddings = graph_embeddings.cpu() # Move to CPU if not already
    
    cos_sim = cosine_similarity(prompt_text_embeddings, text_embeddings).reshape(-1)
    # print("Cos sim: ", cos_sim.shape)
    
    related_node = np.argsort(cos_sim)[::-1][:10]
    # print("related node:", related_node)
    # print("related node shape:", graph_embeddings[related_node[0]].shape)
    return graph_embeddings[related_node[0]]

def get_top_attack_weakness(prompt_related_embeddings, graph_embeddings, top_k1, top_k2):
    attack_size = 203
    weakness_size = 933

    # Ensure prompt_text_embeddings is a CPU tensor if it's used with NumPy
    # prompt_text_embeddings = prompt_text_embeddings.cpu().reshape(1, -1)  # Move to CPU and reshape
    # print(prompt_related_embeddings.shape)
    #graph_embeddings = graph_embeddings.cpu() # Move to CPU if not already
    
    cos_sim_attack = cosine_similarity(prompt_related_embeddings, graph_embeddings[:attack_size]).reshape(-1)
    # print("Cos sim: ", cos_sim_attack.shape)
    cos_sim_weak = cosine_similarity(prompt_related_embeddings, graph_embeddings[attack_size:]).reshape(-1)
    top_attack = np.argsort(cos_sim_attack)[::-1][:top_k1]
    top_weak = np.argsort(cos_sim_weak)[::-1][:top_k2]
    attack_pairs = list(zip(cos_sim_attack[top_attack], top_attack))
    weak_pairs = list(zip(cos_sim_weak[top_weak], top_weak + attack_size))
    return attack_pairs, weak_pairs

def generate_prompt_context(prompt, related_attack, related_weakness, doc_id_to_desc, emb_id_to_doc_id, emb_id_to_tid):
    attack_text = "\n\n".join([f"ATTACK ID: {emb_id_to_tid[str(pos)]}: {doc_id_to_desc[emb_id_to_doc_id[str(pos)]]}" for _, pos in related_attack])
    weakness_text = "\n\n".join([f"CWE ID: {emb_id_to_doc_id[str(pos)]}: {doc_id_to_desc[emb_id_to_doc_id[str(pos)]]}" for _, pos in related_weakness])
    return f"Query: {prompt}\n\nRelated ATT&CK Descriptions:\n\n{attack_text}\n\nRelated CWE Descriptions:\n\n{weakness_text}"

def main(prompt, text_model):
    model_dictn = {"SecureBERT":"pt_SecureBERT","SecBERT":"pt_SecRoBERTa","GPT-2":"pt_gpt2-xl"}
    print("Model Selected: ", text_model)
    if(text_model=="SBERT"):    
        prompt_text_embeddings = retrieve_embeddings(prompt,"SBERT")
        prompt_text_embeddings=prompt_text_embeddings.reshape(1, -1)
        # print("prompt text embedding shape:",prompt_text_embeddings.shape)
        graph_embeddings = np.load(config.OUTPUT_DIR + "embeddings/gpt2-xl/text_spd_embeddings.npy")
        # print("graph embedding shape:",graph_embeddings.shape)
        text_embeddings = np.load(config.OUTPUT_DIR + "embeddings/SBERT/text_embeddings.npy")
        # print("text embedding shape:",text_embeddings.shape)
        
    
        text_embedding_dim = text_embeddings.shape[0]  # Example text embedding dimension
        
        prompt_related_embeddings = get_prompt_related_embeddings(prompt_text_embeddings,text_embeddings,graph_embeddings)
        prompt_related_embeddings = prompt_related_embeddings.reshape(1,-1)
   
    else:
        prompt_text_embeddings = retrieve_embeddings(prompt,text_model)
        
        prompt_text_embeddings=prompt_text_embeddings.reshape(1, -1)
        # print("prompt text embedding shape:",prompt_text_embeddings.shape)
        graph_embeddings = np.load(config.OUTPUT_DIR + "embeddings/{}/text_spd_embeddings_mse_alpha0.5.npy".format(model_dictn[text_model]))
        encoder_model_path = config.OUTPUT_DIR + "embeddings/{}/encoder_decoder_model.pth".format(model_dictn[text_model])
        # print("graph embedding shape:",graph_embeddings.shape)
        hidden_dims1 = [2048,1024,512]
        hidden_dims2 = [512,256,128]
        encoder = Encoder(prompt_text_embeddings.shape[1], hidden_dims1).to(device)
        decoder = Decoder(hidden_dims2, graph_embeddings.shape[1]).to(device)
        prompt_related_embeddings = generate_graph_emb(prompt_text_embeddings,encoder,decoder,encoder_model_path)
    
    # print("prompt_related_embeddings shape:",prompt_related_embeddings.shape)
    related_attack, related_weakness = get_top_attack_weakness(prompt_related_embeddings, graph_embeddings, 10, 10)
    with open(config.DESCRIPTION_FILE) as fp:
        doc_id_to_desc = json.load(fp)
    with open(config.ATTACK_DIR + 'emb_id_to_doc_id.json') as fp:
        emb_id_to_doc_id = json.load(fp)
    with open(config.ATTACK_DIR + 'emb_id_to_tid.json') as fp:
        emb_id_to_tid = json.load(fp)
    augmented_prompt = generate_prompt_context(prompt, related_attack, related_weakness, doc_id_to_desc, emb_id_to_doc_id,emb_id_to_tid)
    return augmented_prompt

if __name__ == "__main__":
    prompt = sys.argv[1]  # Taking prompt as command-line argument
    text_model = sys.argv[2]
    # print("text_model: ", text_model)
    # prompt = "Can you suggest common weaknesses and vulnerabilities related to the Colonial Pipeline Attack? In May of 2021, a hacker group known as DarkSide gained access to Colonial Pipeline’s network through a compromised VPN password. This was possible, in part, because the system did not have multifactor authentication protocols in place. This made entry into the VPN easier since multiple steps were not required to verify the user’s identity. Even though the compromised password was a “complex password,” malicious actors acquired it as part of a separate data breach."
    print(main(prompt,text_model))
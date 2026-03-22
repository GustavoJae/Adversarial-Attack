import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Enganando a IA: Ataque Adversário")

st.title("🐾 Enganando Modelos de Classificação de Animais")
st.markdown("""
Esta aplicação demonstra como pequenas mudanças imperceptíveis em uma imagem (Ataque Adversário)
podem fazer um modelo de Deep Learning de última geração (ResNet50) classificar um animal de forma totalmente errada.
""")

# --- Carregamento do Modelo e Rótulos (Cache para performance) ---
@st.cache_resource
def load_model():
    # Usamos ResNet50 pré-treinada na ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval() # Modo de avaliação
    return model

@st.cache_resource
def load_labels():
    # Carrega os nomes das 1000 classes da ImageNet
    # O arquivo imagenet_class_index.json é comum em tutoriais de PyTorch
    # Se não tiver, o código abaixo baixa automaticamente
    import urllib.request
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = urllib.request.urlopen(url)
    labels_json = json.load(response)
    # Formato: {idx: [nID, label]} -> {idx: label}
    labels = {int(k): v[1] for k, v in labels_json.items()}
    return labels

model = load_model()
labels = load_labels()

# Classes mais conhecidas para simplificar a escolha do alvo na interface.
COMMON_TARGET_LABEL_CANDIDATES = [
    "tabby",
    "tiger_cat",
    "Persian_cat",
    "Siamese_cat",
    "Egyptian_cat",
    "Chihuahua",
    "pug",
    "beagle",
    "golden_retriever",
    "Labrador_retriever",
    "German_shepherd",
    "Siberian_husky",
    "red_fox",
    "timber_wolf",
    "lion",
    "tiger",
    "leopard",
    "cheetah",
    "brown_bear",
    "polar_bear",
    "giant_panda",
    "zebra",
    "giraffe",
    "African_elephant",
    "hippopotamus",
    "koala",
    "kangaroo",
    "gorilla",
    "chimpanzee",
    "orangutan",
    "meerkat",
]

# --- Transformações de Imagem ---
# Mantemos a proporção da imagem para evitar cortes quadrados na saída.
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Parâmetros de normalização da ImageNet (necessários para reverter depois)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])

norm = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

def denormalize(tensor):
    """Reverte a normalização para exibição da imagem."""
    return tensor * imagenet_std.view(3, 1, 1) + imagenet_mean.view(3, 1, 1)

def prepare_image_for_model(image, max_side=512):
    """Redimensiona apenas se necessário, preservando a proporção original."""
    prepared_image = image.copy()

    if max(prepared_image.size) > max_side:
        prepared_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return prepared_image

def format_label_name(label):
    """Deixa o nome da classe mais legível para a interface."""
    return label.replace("_", " ")

def get_popular_target_labels(labels):
    """Filtra um subconjunto de classes populares e usa fallback se necessário."""
    available_labels = []
    seen_labels = set()

    for candidate in COMMON_TARGET_LABEL_CANDIDATES:
        for idx, label in labels.items():
            if label == candidate and label not in seen_labels:
                available_labels.append((idx, label))
                seen_labels.add(label)
                break

    if available_labels:
        return available_labels

    return sorted(labels.items(), key=lambda x: x[1])

def iterative_fgsm_targeted(image, epsilon, target_class_idx, model, iters=20):
    """
    Ataque Iterativo (I-FGSM): Aplica pequenos passos repetidamente.
    """
    perturbed_image = image.clone().detach()
    alpha = epsilon / iters  # O tamanho de cada pequeno passo
    
    for i in range(iters):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        
        target = torch.tensor([target_class_idx])
        loss = F.cross_entropy(output, target)
        
        model.zero_grad()
        loss.backward()
        
        # Movemos a imagem na direção do alvo (minimiza a perda do alvo)
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image.detach() - alpha * data_grad.sign()
        
        # Garante que a imagem não se afaste demais da original (Projeção)
        # Mantém a mudança dentro do limite de epsilon total
        delta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + delta, -2.5, 2.5)
        
    return perturbed_image

# --- Função para Obter Predições e Plotar Distribuição ---
def get_prediction_data(image_tensor, model, labels, top_k=5):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
    top_prob, top_catid = torch.topk(probabilities, top_k)
    
    predict_data = []
    for i in range(top_prob.size(0)):
        predict_data.append((labels[top_catid[i].item()], top_prob[i].item()))
    
    return predict_data

def plot_probabilities(predict_data, title):
    labels_p = [x[0] for x in predict_data]
    probs = [x[1] for x in predict_data]
    
    fig, ax = plt.subplots(figsize=(4, 3))
    y_pos = np.arange(len(labels_p))
    ax.barh(y_pos, probs, align='center', color='#4CAF50')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_p)
    ax.invert_yaxis()  # Maior probabilidade no topo
    ax.set_xlabel('Probabilidade')
    ax.set_title(title)
    ax.set_xlim(0, 1.1)
    
    # Adicionar labels de porcentagem nas barras
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i + .1, f"{v*100:.1f}%", fontsize=8)
        
    plt.tight_layout()
    return fig

# --- Barra Lateral (Sidebar) para Inputs ---
st.sidebar.header("1. Upload da Imagem")
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem de animal...", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.header("2. Configuração do Ataque")

# Seleção da classe alvo com foco em classes mais conhecidas
popular_target_labels = get_popular_target_labels(labels)
label_display_names = [format_label_name(label) for _, label in popular_target_labels]

default_target_label = "golden_retriever"
default_index = next(
    (i for i, (_, label) in enumerate(popular_target_labels) if label == default_target_label),
    0,
)

target_label_display = st.sidebar.selectbox(
    "Escolha a classe alvo ('falsa'):",
    label_display_names,
    index=default_index,
    help="A lista foi filtrada para mostrar apenas classes mais conhecidas.",
)
target_class_idx, target_label_name = popular_target_labels[label_display_names.index(target_label_display)]

# Slider de Epsilon (intensidade da mudança)
# Valores comuns para FGSM na ImageNet: 0.007 (1/255), 0.02, 0.03
epsilon = st.sidebar.slider("Intensidade do Ataque (ε - Epsilon):", min_value=0.0, max_value=0.3, value=0.02, step=0.001, help="Quanto maior o valor, mais visível o ruído, mas maior a chance de erro.")

# Botão para executar
run_attack = st.sidebar.button("Executar Ataque 🚀")

# --- Área Principal de Exibição ---
if uploaded_file is not None:
    # Carregar e pre-processar a imagem original
    image = Image.open(uploaded_file).convert('RGB')
    image_for_model = prepare_image_for_model(image)
    image_tensor_raw = preprocess(image_for_model) # Sem normalização ainda
    image_tensor_norm = norm(image_tensor_raw).unsqueeze(0) # Com normalização e batch dim
    
    # Criar colunas para Antes e Depois
    col1, col2 = st.columns(2)

    # --- COLUNA 1: ORIGINAL ---
    with col1:
        st.subheader("🖼️ Imagem Original")
        st.image(image_for_model, use_container_width=True)

        if image_for_model.size != image.size:
            st.caption(
                f"Imagem redimensionada internamente de {image.size[0]}x{image.size[1]} "
                f"para {image_for_model.size[0]}x{image_for_model.size[1]} para acelerar o processamento."
            )
        
        with st.spinner('Classificando imagem original...'):
            orig_predict_data = get_prediction_data(image_tensor_norm, model, labels)
            st.pyplot(plot_probabilities(orig_predict_data, "Top 5 Predições (Original)"))

    # --- EXECUÇÃO DO ATAQUE ---
    if run_attack:
        with st.spinner(f'Gerando imagem adversária (Alvo: {target_label_name})...'):
            # Realizar o ataque (FGSM Targeted)
            adv_tensor_norm = iterative_fgsm_targeted(image_tensor_norm, epsilon, target_class_idx, model)
            
            # Denormalizar e converter de volta para imagem PIL para exibição
            adv_tensor_denorm = denormalize(adv_tensor_norm.squeeze(0))
            adv_tensor_denorm = torch.clamp(adv_tensor_denorm, 0, 1) # Clamp final para garantir [0, 1]
            adv_image_pil = transforms.ToPILImage()(adv_tensor_denorm)
            
            # Obter predições da imagem adversária
            adv_predict_data = get_prediction_data(adv_tensor_norm, model, labels)
            
        # --- COLUNA 2: ADVERSÁRIA ---
        with col2:
            st.subheader(f"👿 Imagem Adversária (ε={epsilon})")
            st.image(adv_image_pil, use_container_width=True)
            
            # Verificar se o ataque funcionou (se a classe alvo está no topo)
            top_pred_adv = adv_predict_data[0][0]
            if top_pred_adv == target_label_name:
                st.success(f"🎉 Sucesso! O modelo agora acha que é um(a) '{top_pred_adv}'.")
            else:
                st.warning(f"O modelo ficou confuso, mas a classe top ainda é '{top_pred_adv}'. Tente aumentar ε.")
                
            st.pyplot(plot_probabilities(adv_predict_data, "Top 5 Predições (Adversária)"))
            
            # --- Visualização do Ruído (Opcional, mas didático) ---
            with st.expander("Ver o 'Ruído' adicionado (amplificado)"):
                # Diferença entre a original e a adversária (ambas denormalizadas e clampadas)
                # Para visualização, normalizamos a diferença para ocupar todo o espectro de cor
                noise = adv_tensor_denorm - image_tensor_raw
                noise_min = noise.min()
                noise_max = noise.max()
                if noise_max > noise_min:
                    noise = (noise - noise_min) / (noise_max - noise_min)
                else:
                    noise = torch.zeros_like(noise)
                noise_image_pil = transforms.ToPILImage()(noise)
                st.image(noise_image_pil, use_container_width=True, caption="Este é o padrão matemático invisível que engana a rede.")

else:
    st.info("👈 Carregue uma imagem na barra lateral para começar.")
    # Imagem de exemplo ou placeholder
    st.image("https://images.unsplash.com/photo-1560807707-8cc77767d783?q=80&w=600", caption="Exemplo: Um filhote de cachorro.", width=300)
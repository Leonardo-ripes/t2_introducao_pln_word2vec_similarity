
import re
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

def remove_html_tags(text: str | None) -> str:
	"""Remove tags HTML de uma string usando regex."""
	if not text:
		return ""
	return re.sub(r"<[^>]*>", "", text)

def remove_quebras_de_linha(text: str | None) -> str:
    """Remove quebras de linha de uma string usando regex."""
    if not text:
        return ""
    return re.sub(r"\n+", " ", text).strip()

def remover_acentos_pontuacao(text: str | None) -> str:
    """Remove acentos e pontuação de uma string usando regex."""
    if not text:
        return ""
    # Remove acentos
    text_sem_acentos = re.sub(r'[áàâãäå]', 'a', text)
    text_sem_acentos = re.sub(r'[éèêë]', 'e', text_sem_acentos)
    text_sem_acentos = re.sub(r'[íìîï]', 'i', text_sem_acentos)
    text_sem_acentos = re.sub(r'[óòôõö]', 'o', text_sem_acentos)
    text_sem_acentos = re.sub(r'[úùûü]', 'u', text_sem_acentos)
    text_sem_acentos = re.sub(r'[ç]', 'c', text_sem_acentos)
    # Remove indicadores ordinais e símbolo de grau (ex.: n°, nº, 1ª)
    text_sem_acentos = re.sub(r'[°ºª]', '', text_sem_acentos)
    # Remove pontuação
    texto_limpo = re.sub(r'[^\w\s]', '', text_sem_acentos)
    return texto_limpo

def remove_numeros(text: str | None) -> str:
    """Remove números de uma string usando regex."""
    if not text:
        return ""
    return re.sub(r'\d+', '', text)

def remove_datas(text: str | None) -> str:
    """Remove datas de uma string usando regex."""
    if not text:
        return ""
    # Remove datas no formato dd/mm/yyyy ou dd-mm-yyyy
    return re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', '', text)

def normaliza_texto(text: str | None) -> str:
    """Normaliza o texto removendo tags HTML, quebras de linha, acentos e pontuação."""
    if not text:
        return ""
    texto_sem_tags = remove_html_tags(text)
    texto_formatado = remove_quebras_de_linha(texto_sem_tags)
    texto_sem_acentos_pontuacao = remover_acentos_pontuacao(texto_formatado)
    texto_sem_numeros = remove_numeros(texto_sem_acentos_pontuacao)
    texto_sem_datas = remove_datas(texto_sem_numeros)
    texto_final = texto_sem_datas.lower()
    return texto_final

def filtrar_tokens_raros(sentencas_tokenizadas, min_freq=5):
    """
        Filtra tokens raros de uma lista de sentenças tokenizadas, 
        mantendo apenas aqueles que aparecem com frequência mínima.
        Args:
            - sentencas_tokenizadas: Lista de sentenças, onde cada sentença é uma lista de tokens.
            - min_freq: Frequência mínima para um token ser mantido.
        Returns:
            - sentencas_filtradas: Lista de sentenças com tokens raros removidos.
            - frequencias: Dicionário com a frequência de cada token.
    """
    frequencias = Counter(
        token
        for sentenca in sentencas_tokenizadas
        for token in sentenca
    )

    sentencas_filtradas = [
        [token for token in sentenca if frequencias[token] >= min_freq]
        for sentenca in sentencas_tokenizadas
    ]

    return sentencas_filtradas, frequencias

def construir_vocabulario(sentencas_tokenizadas, vocab_size=2000, min_freq=5):
    """
        Constrói um vocabulário a partir de uma lista de sentenças tokenizadas,
        limitando o tamanho do vocabulário e filtrando tokens raros.
        Args:
            - sentencas_tokenizadas: Lista de sentenças, onde cada sentença é uma lista de tokens.
            - vocab_size: Tamanho máximo do vocabulário (excluindo o token <UNK>).
            - min_freq: Frequência mínima para um token ser incluído no vocabulário.
        Returns:
            - vocab: Dicionário mapeando tokens para IDs, incluindo o token <UNK>.
            - id2word: Dicionário mapeando IDs para tokens.
    """
    frequencias = Counter(
        token
        for sentenca in sentencas_tokenizadas
        for token in sentenca
    )

    tokens_ordenados = [
        token for token, freq in frequencias.most_common()
        if freq >= min_freq
    ]

    tokens_ordenados = tokens_ordenados[:vocab_size]

    vocab = {"<UNK>": 0}
    for idx, token in enumerate(tokens_ordenados, start=1):
        vocab[token] = idx

    id2word = {idx: word for word, idx in vocab.items()}

    return vocab, id2word

def gerar_pares_skipgram(sentencas_tokenizadas, vocab, window_size=3):
    """
        Gera pares de palavras para o modelo Skip-Gram a partir de uma lista de sentenças tokenizadas.
        Args:
            - sentencas_tokenizadas: Lista de sentenças, onde cada sentença é uma lista de tokens.
            - vocab: Dicionário mapeando tokens para IDs.
            - window_size: Tamanho da janela de contexto (número de palavras à esquerda e
                            à direita da palavra central).
        Returns:
            - pares: Lista de tuplas (palavra_central_id, palavra_contexto_id).
    """
    pares = []

    for sentenca in sentencas_tokenizadas:
        ids = [vocab.get(token, vocab["<UNK>"]) for token in sentenca]

        for i, centro in enumerate(ids):
            inicio = max(0, i - window_size)
            fim = min(len(ids), i + window_size + 1)

            for j in range(inicio, fim):
                if j != i:
                    contexto = ids[j]
                    pares.append((centro, contexto))

    return pares

def cosseno(v1, v2):
    """
        Calcula a similaridade de cosseno entre dois vetores.
        Args:
            - v1: Primeiro vetor (array numpy).
            - v2: Segundo vetor (array numpy).
        Returns:
            - Similaridade de cosseno entre os dois vetores, um valor entre -1 e 1.
    """
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den == 0:
        return 0.0
    return float(np.dot(v1, v2) / den)

def score_para_grau(score_01):
    """
        Converte um score de similaridade (entre 0 e 1) para um grau de similaridade em incrementos de 0.25.
        Args:
            - score_01: Score de similaridade entre 0 e 1.
        Returns:
            - Grau de similaridade em incrementos de 0.25.
    """
    if score_01 < 0.125:
        return 0.0
    elif score_01 < 0.375:
        return 0.25
    elif score_01 < 0.625:
        return 0.5
    elif score_01 < 0.875:
        return 0.75
    return 1.0

def token_para_id(vocab, token):
    """
        Converte um token para seu ID usando o vocabulário, retornando 0 para tokens desconhecidos.
        Args:
            - vocab: Dicionário mapeando tokens para IDs.
            - token: Token a ser convertido.
        Returns:
            - ID do token, ou 0 se o token não estiver no vocabulário.
    """
    return vocab.get(token, 0)

class SkipGramPairsDataset(Dataset):
    def __init__(self, dataframe):
        self.centros = dataframe["centro"].astype(int).tolist()
        self.contextos = dataframe["contexto"].astype(int).tolist()

    def __len__(self):
        return len(self.centros)

    def __getitem__(self, idx):
        return self.centros[idx], self.contextos[idx]



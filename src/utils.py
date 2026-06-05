
import re

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
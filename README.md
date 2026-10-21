# Agentic RAG: Filtragem de Relevância com Agente Laya

Este projeto demonstra a evolução do RAG (Retrieval-Augmented Generation) tradicional para uma arquitetura **RAG Agêntico**. Ele introduz um "Agente de Qualidade" (Laya) que atua como juiz entre o banco vetorial (Qdrant) e o LLM gerador, descartando conteúdo obsoleto antes que ele contamine a resposta ao cliente final.

## 🎯 O Problema vs. A Solução

**O Problema (Naive RAG):**
Bases de conhecimento mudam. Quando uma regra é atualizada, a versão antiga e a nova são semanticamente quase idênticas, e a busca vetorial devolve **as duas**. O LLM recebe instruções contraditórias, sem saber qual é a vigente. No nosso experimento, o Llama 3.1 aplicou a regra de 2021, já substituída, e negou ao cliente um direito que a política atual garante.

**A Solução (Agentic RAG):**
Desacoplamos a recuperação da geração. Após o Qdrant retornar os Top-K resultados, um modelo pequeno e especializado (`laya:322m`) classifica **qual regra** cada chunk define. Dentro de cada regra, só a importação mais recente é considerada relevante; as versões antigas são descartadas antes de chegar ao LLM final (`Llama 3.1`).

| Pergunta | Naive RAG | Agentic RAG |
|---|---|---|
| Laptop com caixa aberta | ❌ Aplica a regra de 2021 ("perdeu o direito") | ✅ Regra de 2024: 30 dias, caixa aberta aceita |
| Notebook já ligado há 2 semanas | ❌ "Prazo de 7 dias ultrapassado" | ✅ Pode devolver |
| Creatina com pote aberto | ❌ "Não podemos aceitar" | ✅ Até 15 dias, se restar metade |
| Controle: sofá (regra não alterada) | ✅ | ✅ (regra de 2021 mantida) |

## 🏗️ Arquitetura

O fluxo de dados segue 4 etapas principais:
1. **Ingestão:** Dois documentos importados em datas diferentes (política completa de 2021 e atualização **parcial** de 2024). A data fica no payload do Qdrant, não no texto. Cada seção vira um chunk.
2. **Retrieval Bruto:** O usuário faz uma pergunta, e o Qdrant retorna os top 5 resultados usando similaridade vetorial.
3. **Triagem Agêntica (Guardrail):** A Laya classifica cada resultado em uma categoria de regra (pergunta do tipo `choice`). Para cada categoria, versões com data de importação anterior à mais recente são podadas.
4. **Geração Limpa:** O contexto validado é enviado ao LLM de geração final para construir a resposta ao usuário.

### 🧠 Por que a Laya classifica, em vez de dar uma nota de relevância?
A primeira versão usava `noul` ("este texto é relevante para a pergunta?") e obteve 0.99–1.00 para **todos** os chunks, aprovando 5 de 5. Testamos várias formulações que comparam pergunta e trecho, e em nenhuma existe um threshold que separe relevantes de irrelevantes. Já a classificação de um único trecho em categorias acerta todos os chunks da base. O notebook 03 documenta esse experimento.

### 🧩 Estratégia de Chunking (1 seção = 1 chunk)
* **Triagem Cirúrgica:** Para descartar versões obsoletas, cada chunk precisa conter exatamente uma regra. Por isso o texto é primeiro dividido por seção.
* **Fallback seguro:** Seções acima de 1000 caracteres ainda passam pelo `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=100`), mantendo a continuidade semântica.

## 🛠️ Tecnologias Utilizadas

* **Orquestração:** Docker & Docker Compose (com profiles para CPU, Nvidia e AMD GPU)
* **Vector Database:** Qdrant
* **LLM Provider (Local):** Ollama (**v0.35.0 ou superior**, necessário para a API `systemone`)
* **Modelos:** `laya:322m-multilingual-bf16` (Juiz), `llama3.1` (Gerador), `nomic-embed-text-v2-moe` (Embeddings)
* **Framework:** LangChain (text splitters), Jupyter Notebooks

## 🚀 Como Executar o Projeto

Este projeto foi desenhado para rodar 100% localmente via Docker, suportando diferentes configurações de hardware.

### 1. Subir a Infraestrutura (Docker)
Escolha o profile correspondente ao seu hardware:

```bash
# Para máquinas com GPU Nvidia:
docker compose --profile gpu-nvidia up -d

# Para máquinas com GPU AMD:
docker compose --profile gpu-amd up -d

# Para máquinas usando apenas CPU (ou Mac):
docker compose --profile cpu up -d
```

### 2. Preparar o ambiente Python

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 3. Executar os notebooks em ordem
1. `notebooks/01_ingestao_de_dados.ipynb`: cria a coleção e importa as duas versões da política.
2. `notebooks/02_naive_rag_o_problema.ipynb`: mostra o LLM aplicando a regra obsoleta.
3. `notebooks/03_agentic_rag_a_solucao.ipynb`: aplica o agente Laya e compara os dois pipelines em lote.

## 💡 Destaques de Engenharia

* **Eficiência Computacional:** A triagem roda em um SLM de 322M parâmetros (`laya:322m`), muito mais barato do que delegar o julgamento ao modelo de 8B.
* **Cada modelo no seu papel:** A Laya classifica, a data de importação desempata e o Llama só escreve. A decisão final de descarte é uma regra determinística e auditável no código.
* **Fail-open com confiança mínima:** Trechos que a Laya não classifica com confiança (`CONFIANCA_MINIMA`) nunca são descartados, para não perder uma regra vigente.
* **Comparação justa:** Notebooks 02 e 03 usam o mesmo Top-K, o mesmo prompt e `temperature=0`. A única variável do experimento é o filtro.
* **Infraestrutura Escalável:** A arquitetura do `docker-compose.yml` faz uso de *anchors* e *profiles* dinâmicos, permitindo que a mesma stack suba em hardwares diferentes (CPU, Nvidia, AMD) sem alterar uma única linha de código.

# Detecção de Fake News com BERT

## Descrição

Este projeto foi desenvolvido para a disciplina de Fundamentos de Ciência de Dados do curso de Análise e Desenvolvimento de Sistemas (ADS) da UNIFACEMA.

O objetivo é desenvolver um sistema capaz de classificar notícias como verdadeiras ou falsas utilizando Processamento de Linguagem Natural (PLN) e Redes Neurais Profundas baseadas na arquitetura Transformer.

Foi utilizada a técnica de Fine-Tuning do modelo BERT para realizar a classificação de notícias presentes em uma base de dados pública de Fake News.

---

## Objetivo

Treinar um modelo de aprendizado profundo capaz de identificar padrões linguísticos associados a notícias falsas e verdadeiras, auxiliando no combate à desinformação digital.

---

## Tecnologias Utilizadas

- Python 3.12
- PyTorch
- Transformers (Hugging Face)
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Arquitetura Utilizada

O modelo utilizado foi:

**BERT (Bidirectional Encoder Representations from Transformers)**

Modelo pré-treinado baseado na arquitetura Transformer e ajustado através da técnica de Fine-Tuning para classificação binária de textos.

---

## Estrutura do Projeto

```text
TDE-NPL/
│
├── data/
│   ├── raw/
│   │   ├── Fake.csv
│   │   └── True.csv
│   └── processed/
│
├── models/
│
├── results/
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── confusion_matrix_plot.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

Foi utilizada uma base pública contendo notícias verdadeiras e falsas.

Arquivos:

- Fake.csv
- True.csv

Após o pré-processamento, os dados foram combinados em um único conjunto contendo:

- Texto da notícia
- Classe (Fake ou True)

Dataset utilizado:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Etapas do Projeto

### 1. Pré-processamento

Arquivo:

```bash
python src/preprocess.py
```

Responsável por:

- Carregar os arquivos CSV
- Criar os rótulos das classes
- Concatenar os datasets
- Embaralhar os registros
- Gerar o dataset final

---

### 2. Treinamento

Arquivo:

```bash
python src/train.py
```

Responsável por:

- Tokenização dos textos
- Divisão treino/teste
- Fine-Tuning do modelo BERT
- Salvamento do modelo treinado

---

### 3. Avaliação

Arquivo:

```bash
python src/evaluate.py
```

Responsável por calcular:

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusão

---

### 4. Visualização

Arquivo:

```bash
python src/confusion_matrix_plot.py
```

Responsável por gerar:

- Gráfico da Matriz de Confusão

---

## Resultados Obtidos

### Métricas

| Métrica | Resultado |
|----------|----------|
| Accuracy | 99,95% |
| Precision | 100,00% |
| Recall | 99,90% |
| F1-Score | 99,95% |

---

## Matriz de Confusão

| Classe Real | Fake | Verdadeira |
|-------------|------|------------|
| Fake | 1045 | 0 |
| Verdadeira | 1 | 954 |

O modelo classificou corretamente 1999 das 2000 notícias avaliadas.

---

## Metodologia

Foi utilizada a técnica de Transfer Learning através do Fine-Tuning do modelo BERT pré-treinado.

O treinamento foi realizado utilizando uma amostra de 10.000 notícias do conjunto de dados, sendo:

- 80% para treinamento
- 20% para teste

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/Franciele-Santos-Silva/TDE-Ciencia-Dados.git
cd TDE-Ciencia-Dados
```

### 2. Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o Dataset

Baixe o dataset em:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Após o download, coloque os arquivos:

- Fake.csv
- True.csv

na pasta:

```text
data/raw/
```

### 5. Executar o projeto

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/confusion_matrix_plot.py
```

---

## Conclusão

Os resultados demonstraram que modelos baseados em Transformers possuem elevada capacidade para tarefas de classificação textual.

O modelo alcançou 99,95% de acurácia, evidenciando sua eficácia na detecção de Fake News.
Instituição:

**UNIFACEMA – Centro Universitário de Ciências e Tecnologia do Maranhão**

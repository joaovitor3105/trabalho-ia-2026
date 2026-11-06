# Trabalho 2 — Aprendizado de Máquina e Computação Natural

Trabalho Prático da disciplina **G05IART0.02 - Inteligência Artificial**  
**CEFET-MG — Campus Divinópolis** (Semestre 2026/2)  
**Professor:** Tiago Alves de Oliveira  
**Entrega Final:** 03/12/2026 — **Valor:** 20,0 pontos

---

## 👥 Identificação da Equipe

| Integrante | GitHub | E-mail / Contato | Atribuições Principais |
| :--- | :--- | :--- | :--- |
| **João Vitor Lobato Romualdo** | [@joaovitor3105](https://github.com/joaovitor3105) | `joaovitorlobatoromualdo16@gmail.com` | EDA (Parte 1), Ensembles (Random Forest, AdaBoost), Avaliador Único + Algoritmo Genético (Parte 2), Protocolo Experimental e Relatório |
| **Ahmed Amer Hachem** | [@HachemAhmed](https://github.com/HachemAhmed) | `ahmedahachem@gmail.com` | Pipeline sem *Data Leakage*, Modelos Clássicos (Árvore, KNN, SVM), PSO Binário e CLONALG (Parte 2), Testes e Relatório |

---

## 📌 Visão Geral do Projeto

Este repositório reúne duas investigações computacionais integradas:

1. **Parte 1 — Aprendizado Supervisionado e Ensembles (8,0 pts):** Análise exploratória, pré-processamento sem vazamento de dados (*data leakage*) via `Pipeline`/`ColumnTransformer` e comparação experimental sob validação cruzada estratificada (5 folds) de cinco classificadores obrigatórios (**Árvore de Decisão**, **KNN**, **SVM**, **Random Forest** e **AdaBoost**) aplicados à base **Breast Cancer Wisconsin (Diagnostic)** do UCI Machine Learning Repository.
2. **Parte 2 — Computação Natural / Meta-heurísticas (8,0 pts):** Resolução do **Problema 13 da lista oficial (*Seleção de features em um dataset usando uma string binária para indicar atributos*)** com implementação **do zero** de três algoritmos bioinspirados:
   - **Algoritmo Genético (AG)**
   - **Inteligência de Enxame — PSO Binário (BPSO via função sigmoide)**
   - **Sistema Imunológico Artificial — Seleção Clonal (CLONALG)**
3. **Parte 3 — Comparação Experimental, Documentação e Reprodutibilidade (4,0 pts):** Avaliação estatística em **10 execuções independentes** com sementes registradas e **orçamento idêntico de avaliações da função objetivo**, curvas de convergência e relatório técnico em PDF.

---

## 🔬 Registro da Base de Dados (Seção 3.1)

- **Nome da Base:** *Breast Cancer Wisconsin (Diagnostic)*
- **Origem:** [UCI Machine Learning Repository — ID 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (`DOI: 10.24432/C5DW2B`)
- **Autoria:** W. H. Wolberg, W. N. Street e O. L. Mangasarian (University of Wisconsin, 1995)
- **Licença:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Instâncias e Atributos:** 569 instâncias e 30 atributos preditores numéricos reais contínuos (média, erro padrão e pior valor de 10 medidas morfológicas de núcleos celulares em exames FNA).
- **Variável-Alvo e Tarefa:** Classificação binária do diagnóstico (`M` = Maligno / `1`, com 212 amostras [37,26%]; `B` = Benigno / `0`, com 357 amostras [62,74%]).
- **Justificativa Técnica da Escolha:**
  1. **Multicolinearidade e sensibilidade a escala:** A presença de 30 variáveis contínuas em escalas muito distintas (ex.: `area_worst` na casa de $10^3$ vs. `smoothness_se` na casa de $10^{-3}$) e fortemente correlacionadas (ex.: `radius`, `perimeter` e `area`) permite demonstrar de forma clara o efeito do escalonamento (`StandardScaler`) em **KNN** e **SVM**, bem como comparar a poda/divisão de **Árvores** e métodos **Ensemble** (*bagging* vs. *boosting*).
  2. **Integração direta com o Problema 13 (Parte 2):** O espaço de busca de subconjuntos de atributos possui $2^{30} \approx 1,07 \times 10^9$ combinações possíveis, tornando inviável a busca exaustiva e justificando plenamente o emprego de meta-heurísticas (AG, PSO Binário e CLONALG) para encontrar subconjuntos compactos de atributos que preservem ou melhorem o F1-macro de validação.

---

## 🧬 Proposta Técnica de Escopo — Parte 2 (Seção 8 e Apêndice A/B)

### 1. Identificação Oficial do Problema
- **Número e Nome (Apêndice A):** **13. Seleção de features em um dataset usando uma string binária para indicar atributos.**
- **Objetivo:** Maximizar o escore de validação cruzada ($\text{F1}_{\text{macro}}$) penalizando a quantidade de atributos selecionados.

### 2. Formulação Matemática Comum
- **Variáveis de Decisão e Domínio:** Vetor binário $x = (x_1, x_2, \dots, x_D) \in \{0, 1\}^D$, com $D = 30$ atributos da base de treino.
  - $x_i = 1$: o atributo $i$ é incluído no subconjunto selecionado $S(x) = \{i \in \{1,\dots,D\} \mid x_i = 1\}$.
  - $x_i = 0$: o atributo $i$ é descartado.
- **Restrição de Viabilidade:** Uma solução candidata $x$ é **viável** se seleciona ao menos um atributo:
  $$g(x) = \sum_{i=1}^{D} x_i \ge 1$$
  - **Mecanismo de Reparo (`repair(x)`):** Caso uma operação produza o vetor nulo ($\sum_{i=1}^D x_i = 0$), um índice $k \in \{1, \dots, D\}$ é ativado ($x_k \leftarrow 1$) usando o gerador pseudoaleatório controlado pela semente da execução, garantindo $100\%$ de viabilidade antes da avaliação no classificador.
- **Função Objetivo (Maximização):** Avaliada exclusivamente sobre a partição de **treino** (via validação cruzada estratificada de 5 folds com classificador padronizado e fixo para todos os métodos, evitando qualquer vazamento do conjunto de teste final):
  $$f(x) = \text{F1}_{\text{macro, CV}}(X_{\text{train}}[:, S(x)], y_{\text{train}}) - \alpha \cdot \frac{\sum_{i=1}^{D} x_i}{D}$$
  onde $\alpha = 0,05$ pondera o compromisso entre qualidade preditiva e parcimônia (redução de dimensionalidade). O avaliador único reporta separadamente o **F1-CV bruto**, a **fração de features ($|S(x)|/D$)**, o **fitness combinado $f(x)$** e o **status de viabilidade**.

### 3. Representação e Operadores dos Três Métodos
1. **Algoritmo Genético (`src/optimization/genetic.py`):**
   - *Codificação:* Cromossomo binário direto $x \in \{0,1\}^{30}$.
   - *Seleção:* Torneio binário ($k=2$).
   - *Cruzamento:* Uniforme / 2 pontos com probabilidade $p_c = 0,85$.
   - *Mutação:* Inversão de bits (*bit-flip*) com probabilidade $p_m = 1/D \approx 0,033$ por gene.
   - *Substituição:* Geracional com elitismo preservando os $e = 2$ melhores indivíduos.
2. **Inteligência de Enxame — PSO Binário (`src/optimization/swarm.py`):**
   - *Codificação e Adaptação Discreta:* Cada partícula mantém posição binária $x_i \in \{0,1\}^D$ e velocidade real $v_i \in [-v_{\max}, v_{\max}]^D$ ($v_{\max} = 4,0$).
   - *Atualização de Velocidade:* $v_i^{(t+1)} = w v_i^{(t)} + c_1 r_1 (p_{\text{best}, i} - x_i^{(t)}) + c_2 r_2 (g_{\text{best}} - x_i^{(t)})$.
   - *Regra de Decisão Sigmoide (Apêndice A):* A probabilidade de ativar o atributo $d$ é dada por $S(v_{id}) = \frac{1}{1 + e^{-v_{id}}}$, atualizando $x_{id} = 1$ se $u_{id} < S(v_{id})$ (com $u_{id} \sim U(0,1)$) e $0$ caso contrário, seguido de `repair(x)` quando necessário.
3. **Sistema Imunológico Artificial — CLONALG (`src/optimization/immune.py`):**
   - *População de Anticorpos:* $N$ cadeias binárias $x \in \{0,1\}^D$; afinidade diretamente proporcional a $f(x)$.
   - *Seleção e Clonagem:* Seleção dos $n_1$ melhores anticorpos e geração de clones proporcional ao ranking de afinidade.
   - *Hipermutação Somática:* Taxa de inversão de bits **inversamente proporcional à afinidade** ($p_{\text{mut}}(\text{rank}) = p_{\min} + (p_{\max} - p_{\min}) \cdot \frac{\text{rank} - 1}{n_1 - 1}$), preservando soluções de alta afinidade com pequenas perturbações e explorando mais intensamente clones de menor afinidade, seguida de `repair(x)`.
   - *Diversificação:* Substituição dos $d$ anticorpos de menor afinidade por novos receptores aleatórios viáveis a cada geração.

### 4. Protocolo Experimental e Orçamento Comum
- **Orçamento de Avaliações da Função Objetivo:** Fixado em **$B = 1.000$ avaliações de `evaluate()`** por execução para cada um dos três algoritmos (contabilizando inicialização, filhos, partículas e clones mutados).
- **Execuções Independentes:** **10 execuções** por algoritmo com sementes registradas (`seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`).
- **Referências (*Baselines*):** (i) Modelo com todas as 30 features ($x = \vec{1}$); (ii) Busca Aleatória Viável com o mesmo orçamento $B$.

---

## 📂 Estrutura do Repositório

```text
trabalho-ia-2026/
├── README.md
├── requirements.txt
├── Makefile
├── data/
│   ├── README.md                 # Origem, licença, atributos e download
│   ├── raw/                      # Dados brutos da UCI (wdbc)
│   └── processed/                # Partições processadas
├── notebooks/
│   └── eda.ipynb                 # Análise Exploratória de Dados (Parte 1)
├── src/
│   ├── supervised/
│   │   ├── preprocess.py         # Particionamento estratificado e Pipelines
│   │   ├── evaluate.py           # Métricas (Acurácia, P/R/F1 Macro, Matriz de Confusão)
│   │   └── train_models.py       # Treinamento e CV (DT, KNN, SVM, RF, AdaBoost)
│   └── optimization/
│       ├── problem.py            # Avaliador único do Problema 13 e restrições
│       ├── genetic.py            # Algoritmo Genético (do zero)
│       ├── swarm.py              # PSO Binário via Sigmoide (do zero)
│       ├── immune.py             # CLONALG (do zero)
│       └── experiment.py         # Protocolo de 10 execuções e curvas de convergência
├── results/
│   ├── tables/                   # Tabelas CSV/Markdown geradas automaticamente
│   └── figures/                  # Gráficos da EDA, Matrizes de Confusão e Convergência
├── tests/                        # Testes unitários do avaliador, restrições e algoritmos
└── reports/
    └── relatorio.pdf             # Relatório técnico final (6 a 10 páginas)
```

---

## 🚀 Instalação e Execução

```bash
# 1. Instalar dependências
make install

# 2. Executar testes de validação (avaliador, reparo de restrições e reprodutibilidade)
make test

# 3. Executar Parte 1 (Aprendizado Supervisionado e Ensembles)
make part1

# 4. Executar Parte 2 (Experimentos comparativos: AG, PSO Binário e CLONALG)
make part2
```

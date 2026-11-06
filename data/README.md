# Base de Dados — Breast Cancer Wisconsin (Diagnostic)

## 1. Identificação e Origem
- **Nome Oficial:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Repositório de Origem:** [UCI Machine Learning Repository (ID: 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **DOI:** `10.24432/C5DW2B`
- **Autores / Criadores:**
  - Dr. William H. Wolberg (University of Wisconsin Hospitals, Madison)
  - Prof. W. Nick Street (Computer Sciences Department, University of Wisconsin)
  - Prof. Olvi L. Mangasarian (Computer Sciences Department, University of Wisconsin)
- **Licença e Condições de Uso:** Creative Commons Attribution 4.0 International (**CC BY 4.0**). A distribuição dos dados brutos e processados é permitida mediante citação dos autores originais.

## 2. Descrição do Domínio e Coleta
Os atributos foram calculados a partir de imagens digitalizadas de aspirado por agulha fina (*Fine Needle Aspirate* — FNA) de massas mamárias. Eles descrevem características morfológicas dos núcleos celulares presentes na imagem.

Para cada núcleo celular amostrado, dez características reais são avaliadas:
1. `radius` (raio: média das distâncias do centro aos pontos do perímetro)
2. `texture` (textura: desvio-padrão dos valores em escala de cinza)
3. `perimeter` (perímetro nuclear)
4. `area` (área nuclear)
5. `smoothness` (suavidade: variação local nos comprimentos dos raios)
6. `compactness` (compacidade: $\text{perímetro}^2 / \text{área} - 1,0$)
7. `concavity` (concavidade: severidade das porções côncavas do contorno)
8. `concave points` (pontos côncavos: número de porções côncavas do contorno)
9. `symmetry` (simetria)
10. `fractal dimension` (dimensão fractal: *"coastline approximation"* $- 1$)

Para cada uma dessas 10 medidas de base, foram computadas **3 estatísticas por imagem**:
- **Média (`mean`)** — atributos 1 a 10
- **Erro Padrão (`se`)** — atributos 11 a 20
- **Pior Valor / Maior Média dos 3 maiores valores (`worst`)** — atributos 21 a 30

Totalizando **30 atributos preditores numéricos contínuos**.

## 3. Caracterização da Tarefa
- **Tipo de Tarefa:** Classificação supervisionada binária.
- **Número de Instâncias:** 569 amostras.
- **Número de Atributos Preditores:** 30 atributos numéricos reais (além do `ID`, que é descartado por ser apenas identificador de registro).
- **Variável-Alvo (`Diagnosis`):**
  - `M` (*Malignant* — Maligno, codificado como `1`): 212 instâncias ($37,26\%$)
  - `B` (*Benign* — Benigno, codificado como `0`): 357 instâncias ($62,74\%$)
- **Valores Ausentes:** 0 (nenhum valor ausente na versão *Diagnostic* `wdbc.data`).
- **Limitações e Características Críticas:**
  - Forte **multicolinearidade estrutural** entre grupos de variáveis (ex.: `radius`, `perimeter` e `area`, bem como entre `mean` e `worst`), o que impacta modelos baseados em distância (KNN) e margem (SVM), além de tornar a base ideal para o estudo de **Seleção de Features (Problema 13 da Parte 2)**.
  - Desbalanceamento moderado entre classes ($62,7\%$ vs. $37,3\%$), exigindo particionamento estratificado e análise por métricas macro (Precisão Macro, Revocação Macro e F1 Macro) além da acurácia.

## 4. Como Obter / Reproduzir os Dados
Os dados são carregados de maneira 100% reprodutível via script (`src/supervised/preprocess.py`) diretamente do repositório oficial da UCI (`ucimlrepo` / arquivo `wdbc.data`) e salvos em `data/raw/` e `data/processed/`.

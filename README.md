# Tech Challenge - Pós-Graduação em Data Analytics

**Instituição:** FIAP - Turma 2DTAT

![](https://github.com/alexandre-guerra/Analise-IBOV/blob/master/imagem_tech.png)

## POSTECH FIAP - Fase 4 - Data Viz and Production Models

### Título

**Prevendo valores do Petróleo tipo BRENT com algoritmos de machine learning**

---

### Objetivo

Desenvolver um modelo preditivo com dados da série histórica do Petróleo tipo Brent para criar uma série temporal e prever 7 dias do valor de fechamento.

---

### Introdução

No universo financeiro, a previsão de índices de mercado, como o Petróleo tipo Brent, representa um desafio notável, tanto para analistas experientes quanto para modelos preditivos avançados. Este projeto nasce do entendimento de que o mercado é inerentemente complexo e imprevisível, caracterizado por um comportamento que muitas vezes parece caótico e influenciado por uma miríade de fatores.

Primeiramente, é fundamental reconhecer que o mercado não segue padrões lineares ou previsíveis de maneira simples. Ele é influenciado por uma vasta gama de variáveis que vão desde indicadores econômicos macroscópicos até eventos geopolíticos imprevisíveis, passando por mudanças nas políticas governamentais e até mesmo por sentimentos e comportamentos dos investidores, que muitas vezes são reativos e emocionais.

Além disso, a própria natureza do mercado, com sua dinâmica de oferta e demanda, cria um ambiente onde a volatilidade é uma constante. Ações individuais e índices podem experimentar flutuações significativas em curtos períodos de tempo, tornando a tarefa de prever o fechamento diário do Brent extremamente desafiadora.

Neste contexto, o uso de algoritmos de machine learning surge como uma abordagem promissora, pois oferece a possibilidade de identificar padrões ocultos e correlações em grandes conjuntos de dados que seriam difíceis, se não impossíveis, de serem analisados manualmente. No entanto, mesmo essas técnicas sofisticadas não são infalíveis e enfrentam desafios significativos devido à natureza imprevisível do mercado.

Este projeto busca explorar essa complexidade, utilizando um algoritmo XGBoost para construir modelos preditivos. Nosso objetivo é não apenas desenvolver modelos que possam prever o fechamento diário com uma precisão razoável, mas também entender as limitações e desafios enfrentados ao longo deste percurso.

---

### Desenvolvimento do Projeto em Três Fases

1. **Análise exploratória de dados**
   - Descrição da análise realizada
   - Principais insights encontrados

2. **Criação de um modelo preditivo utilizando XGBoost**
   - Implementação do modelo XGBoost
   - Avaliação dos resultados
   - Deploy do modelo em produção

3. **Criação de um Dashboard interativo**
   - Exibir os principais insights

4. **Criação de um VPM do modelo em produção**
   - Criar uma página em Streamlit para apresentar os resultados do modelo

---

### Principais Bibliotecas Utilizadas

Este projeto utiliza várias bibliotecas Python no campo de análise de dados e machine learning, incluindo:

- **Numpy**: Uma biblioteca fundamental para computação científica com Python, utilizada para operações eficientes em arrays multidimensionais.
- **Pandas**: Ferramenta de manipulação e análise de dados de alto desempenho, utilizada para a manipulação de datasets.
- **Plotly**: Biblioteca gráfica para criar visualizações de dados interativas.
- **yfinance**: Usada para baixar dados históricos do mercado financeiro diretamente do Yahoo Finance.
- **Statsmodels**: Fornecendo implementações de várias estatísticas, modelos e testes estatísticos.
- **XGBoost**: Biblioteca otimizada de aumento de gradiente distribuída, projetada para ser altamente eficiente, flexível e portátil.
- **Scikit-Learn**: Empregada para diversas tarefas de machine learning, incluindo o pré-processamento de dados e a avaliação de modelos.

Essas bibliotecas são integradas para coletar dados, processá-los, construir modelos preditivos e avaliar seu desempenho.

---

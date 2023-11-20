# FivePersonalityTestAI
### Aluno: Thiago Bittencourt Santana
![big-five](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/6867bb19-c10c-44de-bb76-85e5afef199e)


Projeto acadêmico que visa desenvolver, através do uso de `IA(Perceptron)`, o teste das cinco grandes personalidades

**Link para o Collab:** [FivePersonalityPerceptron](https://colab.research.google.com/drive/1qgGGpR-c6QHsj73VjWGgkNiHYdmgAngm?usp=sharing)

**Link para o Relatório (.DOC):** [Relatório](https://docs.google.com/document/d/1EsO6cMLImM0q54b9lB6-kMQgfqT168ngymqvUYwIosw/edit?usp=sharing)

---

### Index:
<!--ts-->
   * [Sobre o Teste das Cinco Grandes Personalidades](#sobre-o-teste-das-cinco-grandes-personalidades)
   * [Data Set](#data-set)
   * [Treinamento](#treinamento)
<!--te-->

---

### Sobre o Teste das Cinco Grandes Personalidades:
Os Cinco Grandes Traços de Personalidade, também conhecidos como modelo de cinco fatores (FFM) e modelo OCEAN, são uma taxonomia, ou agrupamento, para traços de personalidade. O teste consiste em 50 perguntas, que de acordo com as respostas, encaixarão a pessoa em um dos cinco grupos: `Abertura a novas experiências; Conscienciosidade; Extroversão; Neuroticismo; Simpatia.`

### Data Set:
- **Nome:** Cinco Grandes Personalidades Teste
- **Link:** [FivePersonalityDB](https://www.kaggle.com/datasets/tunguz/big-five-personality-test)
- **Época de Coleta:** 2016-2018
- **Número de linhas:** 1015341
- **Número de colunas:** 110
- **Colunas:**
> EXT1, EXT2, EXT3, EXT4, EXT5, EXT6, EXT7, EXT8, EXT9, EXT10, EST1, EST2, EST3, EST4, EST5, EST6, EST7, EST8, EST9, EST10, AGR1, AGR2, AGR3, AGR4, AGR5, AGR6, AGR7, AGR8, AGR9, AGR10, CSN1, CSN2, CSN3, CSN4, CSN5, CSN6, CSN7, CSN8, CSN9, CSN10, OPN1, OPN2, OPN3, OPN4, OPN5, OPN6, OPN7, OPN8, OPN9, OPN10, EXT1_E, EXT2_E, EXT3_E, EXT4_E, EXT5_E, EXT6_E, EXT7_E, EXT8_E, EXT9_E, EXT10_E, EST1_E, EST2_E, EST3_E, EST4_E, EST5_E, EST6_E, EST7_E, EST8_E, EST9_E, EST10_E, AGR1_E, AGR2_E, AGR3_E, AGR4_E, AGR5_E, AGR6_E, AGR7_E, AGR8_E, AGR9_E, AGR10_E, CSN1_E, CSN2_E, CSN3_E, CSN4_E, CSN5_E, CSN6_E, CSN7_E, CSN8_E, CSN9_E, CSN10_E, OPN1_E, OPN2_E, OPN3_E, OPN4_E, OPN5_E, OPN6_E, OPN7_E, OPN8_E, OPN9_E, OPN10_E, dateload, screenw, screenh, introelapse, testelapse, endelapse, IPC, country, lat_appx_lots_of_err, long_appx_lots_of_err
- **Sobre o Dataset:**

  O Dataset que utilizei para implementação do perceptron, é o das cinco grandes personalidades: Um Dataset de mais de 1 milhão de linhas, com 110 colunas como entradas, sendo:

  - Da coluna `1 à 50`: **cada uma das perguntas do teste**.
  > (As respostas para cada pergunta são entre 1=Discordo, 3=Neutro, 5=Concordo)
  
  - Da coluna `50 à 100`: **o tempo de resposta do usuário para cada uma das perguntas do teste**
 
  - Da coluna `100 à 110` são apenas metadados sobre a pesquisa
 
  Para o treinamento utilizei apenas as colunas entre `1 e 50` como entrada, pois apenas os níveis de resposta para cada uma das questões é importante para o resultado do teste.

  Entretanto, como pode-se notar do Dataset, ele não consta com uma coluna do resultado do teste. E como o processo de aprendizado no Perceptron precisa ser supervisionado, realizei (com propósito de estudo) ao início do programa um Clustering utilizando o Scikit-Learn (Kmeans), atribuindo ao final, para cada das linhas, um cluster correspondente. E utilizei essa coluna para o processo de treinamento do Perceptron.
  > (Clusters: 0, 1, 2, 3, 4)

  Ao início do programa, realizei a exclusão das colunas indesejadas, e logo após, a exclusão das linhas que continham como respostas às perguntas o valor 0 (pois estão esperadas apenas respostas entre 1 e 5)
---

### Treinamento:
- **Detalhes:**

  Ao iniciar os testes, percebi que utilizar o treinamento com todas as linhas do Dataset levava muito tempo, então reduzi a apenas alguns milhares de dados. Após isso, separei o DS (de acordo com o train/test split) em dados de treino e teste.

- **Testes**

  Tangente Hiperbólica:
  ```python
  o1 = np.tanh(W1.dot(Xb)) 
  ```
  - 1º teste (Tangente Hiperbólica)

    Variáveis:
    ```python
    numEpocas = 100         # Número de épocas.
    q = 100                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.01              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:

    ![Figure_2](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/1d1cabac-7445-45b7-9dc2-432557ef9b95)

  - 2º teste (Tangente Hiperbólica)

    Variáveis:
    ```python
    numEpocas = 200         # Número de épocas.
    q = 500                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.02              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:

    ![Figure_3](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/62dbf903-136e-41b0-b069-d4324f5a39c3)


  - 3º teste (Tangente Hiperbólica)
    
    Variáveis:
    ```python
    numEpocas = 300         # Número de épocas.
    q = 1000                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.04              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:
  
    ![download](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/34d3587a-90b4-4cd2-a793-4fb36648e132)

    ```
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837],
    [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837], [0.99999837]
    
    [4.00000163e+00 2.00000163e+00 2.00000163e+00 1.00000163e+00
     4.00000163e+00 4.00000163e+00 3.00000163e+00 4.00000163e+00
     1.00000163e+00 3.00000163e+00 3.00000163e+00 3.00000163e+00
     1.62862233e-06 3.00000163e+00 1.62862233e-06 4.00000163e+00
     3.00000163e+00 1.62862233e-06 1.62862233e-06 2.00000163e+00
     1.62862233e-06 3.00000163e+00 4.00000163e+00 1.00000163e+00
     1.62862233e-06 2.00000163e+00 4.00000163e+00 2.00000163e+00
     4.00000163e+00 1.62862233e-06 2.00000163e+00 3.00000163e+00
     1.00000163e+00 1.00000163e+00 4.00000163e+00 1.62862233e-06
     2.00000163e+00 4.00000163e+00 4.00000163e+00 3.00000163e+00
     2.00000163e+00 2.00000163e+00 3.00000163e+00 1.62862233e-06
     4.00000163e+00 2.00000163e+00 4.00000163e+00 1.62862233e-06
     1.00000163e+00 2.00000163e+00 3.00000163e+00 1.00000163e+00
     3.00000163e+00 1.62862233e-06 4.00000163e+00 1.62862233e-06
     3.00000163e+00 1.62862233e-06 1.62862233e-06 4.00000163e+00
     1.00000163e+00 1.00000163e+00 3.00000163e+00 3.00000163e+00
     2.00000163e+00 4.00000163e+00 1.62862233e-06 1.00000163e+00
     3.00000163e+00 3.00000163e+00 3.00000163e+00 1.00000163e+00
     3.00000163e+00 4.00000163e+00 1.62862233e-06 3.00000163e+00
     2.00000163e+00 4.00000163e+00 3.00000163e+00 3.00000163e+00
     2.00000163e+00 1.62862233e-06 3.00000163e+00 3.00000163e+00
     3.00000163e+00 2.00000163e+00 1.62862233e-06 4.00000163e+00
     1.00000163e+00 4.00000163e+00 4.00000163e+00 4.00000163e+00
     2.00000163e+00 2.00000163e+00 3.00000163e+00 3.00000163e+00
     3.00000163e+00 1.62862233e-06 4.00000163e+00 1.62862233e-06]
    
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    ```
---
  Sigmoid:
  
  ```python
  def sigmoid(X):
   return 1/(1+np.exp(-X))

  o1 = sigmoid(W1.dot(Xb))
  ```
  - 1º teste (Sigmoid)

    Variáveis:
    ```python
    numEpocas = 100         # Número de épocas.
    q = 100                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.01              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:
    
    ![sigmoid1](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/ed614994-56c4-41cc-a39f-5d0df1fcdd27)

    - 2º teste (Sigmoid)

    Variáveis:
    ```python
    numEpocas = 200         # Número de épocas.
    q = 500                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.02              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:

    ![sigmoid2](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/9a193bc8-4127-4a05-a88c-403408c2ae01)

     - 3º teste (Sigmoid)
    
    Variáveis:
    ```python
    numEpocas = 300         # Número de épocas.
    q = 1000                # Número de padrões treinamento.
    q2 = 100                # Número de padrões testes.
  
    eta = 0.04              # Taxa de aprendizado.
    m = 50                  # Número de neurônios na camada de entrada.
    N = 10                   # Número de neurônios na camada escondida.
    L = 1                   # Número de neurônios na camada de saída.
    ```
    Resultados:

    ![sigmoid3](https://github.com/oThiagoBittencourt/PerceptronBigFivePersonalityTest/assets/106789198/a9cdfb78-3ffc-4b9a-9901-c604697373da)

    ```
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847],
    [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847], [0.99999847]
    
    [4.00000153e+00 2.00000153e+00 2.00000153e+00 1.00000153e+00
     4.00000153e+00 4.00000153e+00 3.00000153e+00 4.00000153e+00
     1.00000153e+00 3.00000153e+00 3.00000153e+00 3.00000153e+00
     1.52519300e-06 3.00000153e+00 1.52519300e-06 4.00000153e+00
     3.00000153e+00 1.52519300e-06 1.52519300e-06 2.00000153e+00
     1.52519300e-06 3.00000153e+00 4.00000153e+00 1.00000153e+00
     1.52519300e-06 2.00000153e+00 4.00000153e+00 2.00000153e+00
     4.00000153e+00 1.52519300e-06 2.00000153e+00 3.00000153e+00
     1.00000153e+00 1.00000153e+00 4.00000153e+00 1.52519300e-06
     2.00000153e+00 4.00000153e+00 4.00000153e+00 3.00000153e+00
     2.00000153e+00 2.00000153e+00 3.00000153e+00 1.52519300e-06
     4.00000153e+00 2.00000153e+00 4.00000153e+00 1.52519300e-06
     1.00000153e+00 2.00000153e+00 3.00000153e+00 1.00000153e+00
     3.00000153e+00 1.52519300e-06 4.00000153e+00 1.52519300e-06
     3.00000153e+00 1.52519300e-06 1.52519300e-06 4.00000153e+00
     1.00000153e+00 1.00000153e+00 3.00000153e+00 3.00000153e+00
     2.00000153e+00 4.00000153e+00 1.52519300e-06 1.00000153e+00
     3.00000153e+00 3.00000153e+00 3.00000153e+00 1.00000153e+00
     3.00000153e+00 4.00000153e+00 1.52519300e-06 3.00000153e+00
     2.00000153e+00 4.00000153e+00 3.00000153e+00 3.00000153e+00
     2.00000153e+00 1.52519300e-06 3.00000153e+00 3.00000153e+00
     3.00000153e+00 2.00000153e+00 1.52519300e-06 4.00000153e+00
     1.00000153e+00 4.00000153e+00 4.00000153e+00 4.00000153e+00
     2.00000153e+00 2.00000153e+00 3.00000153e+00 3.00000153e+00
     3.00000153e+00 1.52519300e-06 4.00000153e+00 1.52519300e-06]
    
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0.]
    ```
---

**Conclusão:**

Pode-se analisar que a função sigmoid se sai melhor com o treinamento com poucos dados. Já quando se é expandido o número de testes (padrões e épocas), a função de ativação da tangente hiperbólica apresenta um melhor resultado.

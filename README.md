# Open The Chests — Notas de Design

## Reward Design

### Problema: Recompensa Esparsa
Para evitar recompensas esparsas, usar **recompensas intermediárias (dense rewards)** que guiam o agente durante o episódio.

### Estratégias de Reward

**Distância por range vs. Euclidiana**
- Range de distância é mais simples e menos custoso computacionalmente que distância euclidiana pura
- Testar as duas é uma boa ablação para comparar velocidade de convergência

**HER — Hindsight Experience Replay**
- Acelera o aprendizado reutilizando episódios falhos como se tivessem alcançado um goal diferente
- O agente aprende algo útil mesmo de trajetórias que não atingiram o objetivo real

**O que incluir**
- Distância do end-effector à caixa-alvo (X, Y, Z)
- Bônus de proximidade por zonas
- Bônus de sucesso ao manter posição

**O que evitar**
- Rewards que incentivem comportamentos errados por proxy
  - Ex: no Pong, o agente pode aprender a seguir a raquete adversária em vez de rastrear a bola
- Penalidades em variáveis que nunca mudam (ex: posição da caixa fixa)

---

## Escolha do Algoritmo

| Abordagem | Algoritmo | Observação |
|---|---|---|
| **Policy-based** | PPO, A2C | Recomendado — lida naturalmente com ações contínuas 3D |
| **Value-based** | DQN | Requer discretização (bucketização) do espaço de ação, o que degrada precisão |

**Por que policy-based é melhor aqui:**
DQN precisa de `argmax` sobre todas as ações possíveis. Em um espaço contínuo 3D, isso exige discretizar as ações em bins, criando um espaço latente de valores muito grande e prejudicando a generalização entre ações próximas.


### PPO (on-policy) vs SAC (off-policy)

Testar os dois com o mesmo budget de `total_timesteps` para comparar velocidade de convergência e qualidade da policy final.

**Hipótese:** SAC converge mais rápido que PPO em número de timesteps, dado que o ambiente é determinístico e o reward é denso.

**Métricas de comparação:**
- Timesteps até convergência (eficiência de amostra)
- Accumulated reward ao longo do treino(manter 1 tipo de reward só pra ficar mais simples, utilisar por range só por ex)
- Distância final ao alvo (`distance_to_target`)
- Visualização e manter constante o output: [rliable](https://github.com/google-research/rliable)



### Propriedade de Markov

A observação do OTC **não é markoviana**: para saber qual caixa abrir, o agente precisa conhecer o histórico de eventos, não apenas o estado atual.

Para tornar o problema markoviano, é necessário incluir os eventos passados na observação — com 3 caixas e 3 eventos possíveis, o espaço de estados relevantes é de 9 combinações.

| Versão | Arquitetura necessária | Motivo |
|---|---|---|
| v0 | MLP (sem memória) | Observação já contém a cor-alvo diretamente |
| v1, v2 | RNN / LSTM / Transformer | A cor-alvo depende de uma sequência de eventos passados |

**Janela mínima de histórico:** 2 a 3 vezes a quantidade de eventos — suficiente para cobrir as combinações relevantes e tornar o estado markoviano.

Observation vector dimension
- total observation dimension: 20 atributos por eventos--> 20x4 = 80 features/eventos, se guardamors ultimos 8 eventos: 160 features
- padding for the begging
- k choose: too small(non markovian) ou too big(slow learning)
# ai_weather
Pesquisa de Mestrado em Ciência da Computação (PPGCC-PUCRS) - "AI weather forecasting: a case study for Brazil"
Aluno: Wolfgang Rowell
Orientador: Lucas Kupssinskü, PhD

Descrição dos arquivos:
- graphcast_demo.ipynb - Versão original do Google Deepmind de demonstração de como carregar dados, gerar gráficos, gerar previsões e treinar o modelo.
- graphcast_LoRA.ipynb - Versão modificada por mim com as seguintes alterações:
  - inclui pip install do jax para cuda12 e o optax (otimizador do JAX) que está sendo usado no treinamento (existe um método nativo no JAX/Haiku para congelar os pesos  `jax.lax.stop_gradient()`)
  - inclui uma célula logo no início com os patches para alterar as classes e funções direto no venv do servidor do colab, que permitisse congelar os parâmetros do enconder e do processor (16 steps);
  - deletei todas células de geração de gráficos e visualização;
  - bypassei todas as seleções manuais de parâmetros desnecessários, deixei apenas 2: a seleção de carregar o dataset com os steps do groundtruth, e mais abaixo a seleção de quantos steps de treino/verificação;
  - a única etapa que está com os pesos liberados para aprender é o MLP do decoder (grid2mesh);

**Tenho utilizado o GPU A100 High RAM, consegui rodar o treino do decoder com até 8 steps e verification 10 steps, com os demais pesos congelados.**

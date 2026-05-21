O trabalho final é um estudo de uma base de dados/problema com redes neurais. Ele é composto pelo desenvolvimento e apresentação do artigo. Por enquanto, esta atividade trata da proposição de trabalho. 

Na proposta é preciso apontar:

- descrição do problema;
- fonte dos dados;
- quem já estudou esses dados (artigos, repositórios etc.);
- quem já estudou os dados da forma que você quer;
- o que você pretende fazer: métodos, análises, avaliações;
- o que de novidade você quer trazer; que análises quer realizar
- qual a fonte dos códigos que você usará ou se irá fazer desenvolvimento próprio.

Sua tarefa é escrever essa proposićão do trabalho tendo em mente as seguintes informacoes:

- O trabalho desenvolvido sera um preditor apostas esportivas paraa partidas do campeonato brasileiro com base nos dados historicos e nas odds calculadas pelas casas de apostas. A fonte de dados foi o site footbal-data.co.uk/brazil.php que funciona como arquivo de resultados de partidas e odds.
- nao site estudos anteriores mas inclua um capitulo para analise na bibliografia. eu manualmente irei relacionar seu texto com os estudos bibliograficos que fiz.
- o trabalho sera desenvolvido em python. 
- sera feita a limpeza e featuree engeneering nos dados, alem do treinamento de dois modelos pra comparacao. um modelo sera um xgboost e outro uma rede neural recorrente
- serao usados dois datasets distintos que serao concatenados. as colunas de fetures disponiveis nos datasets sao:
ano_campeonato	data	rodada	estadio	arbitro	publico	publico_max	time_mandante	time_visitante	tecnico_mandante	tecnico_visitante	colocacao_mandante	colocacao_visitante	valor_equipe_titular_mandante	valor_equipe_titular_visitante	idade_media_titular_mandante	idade_media_titular_visitante	gols_mandante	gols_visitante	gols_1_tempo_mandante	gols_1_tempo_visitante	escanteios_mandante	escanteios_visitante	faltas_mandante	faltas_visitante	chutes_bola_parada_mandante	chutes_bola_parada_visitante	defesas_mandante	defesas_visitante	impedimentos_mandante	impedimentos_visitante	chutes_mandante	chutes_visitante	chutes_fora_mandante	chutes_fora_visitante Country	League	Season	Date	Time	Home	Away	HG	AG	Res	PSCH	PSCD	PSCA	MaxCH	MaxCD	MaxCA	AvgCH	AvgCD	AvgCA	BFECH	BFECD	BFECA	B365CH	B365CD	B365CA

OBS: features terminadas em CH - home team win odd, CD - draft odd, CA - away team win odd.

- esses dados serao limpos, normalizados e por fim combinados nas novas features: 
    1. vitoria do time da casa, vitorias do time de fora, empates no historico do confronto
    2. qntd de vitorias do time da casa e do time de fora no campeonato do ano até a partida em questao
    3. gols pro, gols sofridos e saldo de gols do time da casa e do time de fora no campeonato do ano ate a partida em questao

 - serao analisadas as saidas da rede para classificacao do confronto (time da casa, empate, time de fora) e a regressao de gols para o time da casa e time de fora 
 - alem da analise dos resultados serao analisadas metricas de relevancia para as features presentes a fim de verificar padroes; alem da analise da regra de bolso empirica (quantas linhas por feature ate meu modelo piorar)
 - serao utilizadas taticas de sepraćão de dataset em treino e validacao com 5 inicializacoes diferentes para verificar a estabilidade dos resultados

A divisao de sećões para o relatorio será:

introducao - contexto de apostas esportivas e proposta do trabalho
revisao bibliografica - (em branco apenas com o titulo) - queem já estudou os dados e trabalhos 
conjunto de dados - explicando os dados e seus significados
metodologia - explicnado o que se plneja inicilmente fazer
analise - explicnado o que sera analisado e o q se espera como saida da rede

escreva o documento em latex. evite usaar bullet points, prefira dissertar sobre os temas. evite uma linguagem extremamente rebuscada ou uma fala nao natural. mantenha o discurso academico compativel com um artigo de graduacoa

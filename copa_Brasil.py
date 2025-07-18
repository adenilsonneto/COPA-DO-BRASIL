import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import heapq # Fila de prioridade, essencial para Dijkstra eficiente


class Grafo:

    def __init__(self):
        self.adjacencia = {}

    def inserir_no(self, nome_no):
        if nome_no not in self.adjacencia:
            self.adjacencia[nome_no] = {}

    def inserir_aresta(self, no1, no2, peso):
        if no1 in self.adjacencia and no2 in self.adjacencia:
            self.adjacencia[no1][no2] = peso
            self.adjacencia[no2][no1] = peso

class UnionFind:
   
    def __init__(self, nos):
        self.parent = {no: no for no in nos}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        raiz_i = self.find(i)
        raiz_j = self.find(j)
        if raiz_i != raiz_j:
            self.parent[raiz_j] = raiz_i

def kruskal(grafo: Grafo):

    arestas = []
    nos_visitados = set()
    for no, vizinhos in grafo.adjacencia.items():
        for vizinho, peso in vizinhos.items():
            if (vizinho, no) not in nos_visitados:
                arestas.append((no, vizinho, peso))
                nos_visitados.add((no, vizinho))
    
    arestas.sort(key=lambda item: item[2])

    mst_resultado = []
    nos = list(grafo.adjacencia.keys())
    uf = UnionFind(nos)

    for aresta in arestas:
        no1, no2, peso = aresta
        if uf.find(no1) != uf.find(no2):
            uf.union(no1, no2)
            mst_resultado.append(aresta)
    
    return mst_resultado

def dijkstra(grafo: Grafo, no_inicio, no_fim):

    adj = grafo.adjacencia
    distancias = {no: float('inf') for no in adj}
    predecessores = {no: None for no in adj}
    distancias[no_inicio] = 0
    
    pq = [(0, no_inicio)]

    while pq:
        distancia_atual, no_atual = heapq.heappop(pq)

        if distancia_atual > distancias[no_atual]:
            continue

        if no_atual == no_fim:
            break

        for vizinho, peso in adj[no_atual].items():
            nova_distancia = distancia_atual + peso
            if nova_distancia < distancias[vizinho]:
                distancias[vizinho] = nova_distancia
                predecessores[vizinho] = no_atual
                heapq.heappush(pq, (nova_distancia, vizinho))

    caminho = []
    no = no_fim
    if distancias[no] == float('inf'):
        return [], float('inf')

    while no is not None:
        caminho.append(no)
        no = predecessores[no]
    
    return caminho[::-1], distancias[no_fim]


dados_copa_brasil = [
    # Ano 2021
    {'clube': 'Atlético-MG', 'fase': 'campeão'}, {'clube': 'Athletico-PR', 'fase': 'final'}, {'clube': 'Flamengo', 'fase': 'semifinal'}, {'clube': 'Fortaleza', 'fase': 'semifinal'}, {'clube': 'Grêmio', 'fase': 'quartas'}, {'clube': 'São Paulo', 'fase': 'quartas'}, {'clube': 'Fluminense', 'fase': 'quartas'}, {'clube': 'Santos', 'fase': 'quartas'}, {'clube': 'Bahia', 'fase': 'oitavas'}, {'clube': 'Vasco da Gama', 'fase': 'oitavas'}, {'clube': 'ABC', 'fase': 'oitavas'}, {'clube': 'Remo', 'fase': 'oitavas'}, {'clube': 'Coritiba', 'fase': '3ª fase'}, {'clube': 'Corinthians', 'fase': '3ª fase'}, {'clube': 'Internacional', 'fase': '3ª fase'}, {'clube': 'Palmeiras', 'fase': '3ª fase'}, {'clube': 'Botafogo', 'fase': '3ª fase'},
    # Ano 2022
    {'clube': 'Flamengo', 'fase': 'campeão'}, {'clube': 'Corinthians', 'fase': 'final'}, {'clube': 'Fluminense', 'fase': 'semifinal'}, {'clube': 'São Paulo', 'fase': 'semifinal'}, {'clube': 'Athletico-PR', 'fase': 'quartas'}, {'clube': 'Fortaleza', 'fase': 'quartas'}, {'clube': 'Atlético-MG', 'fase': 'oitavas'}, {'clube': 'Bahia', 'fase': 'oitavas'}, {'clube': 'Botafogo', 'fase': 'oitavas'}, {'clube': 'Ceará', 'fase': 'oitavas'}, {'clube': 'Cruzeiro', 'fase': 'oitavas'}, {'clube': 'Palmeiras', 'fase': 'oitavas'}, {'clube': 'Santos', 'fase': 'oitavas'}, {'clube': 'Coritiba', 'fase': '3ª fase'}, {'clube': 'Remo', 'fase': '3ª fase'},
    # Ano 2023
    {'clube': 'São Paulo', 'fase': 'campeão'}, {'clube': 'Flamengo', 'fase': 'final'}, {'clube': 'Corinthians', 'fase': 'semifinal'}, {'clube': 'Grêmio', 'fase': 'semifinal'}, {'clube': 'Palmeiras', 'fase': 'quartas'}, {'clube': 'Bahia', 'fase': 'quartas'}, {'clube': 'Athletico-PR', 'fase': 'quartas'}, {'clube': 'Atlético-MG', 'fase': 'oitavas'}, {'clube': 'Internacional', 'fase': 'oitavas'}, {'clube': 'Fluminense', 'fase': 'oitavas'}, {'clube': 'Cruzeiro', 'fase': 'oitavas'}, {'clube': 'Fortaleza', 'fase': 'oitavas'}, {'clube': 'Santos', 'fase': 'oitavas'}, {'clube': 'Botafogo', 'fase': 'oitavas'}, {'clube': 'Sport', 'fase': 'oitavas'}, {'clube': 'ABC', 'fase': '3ª fase'}, {'clube': 'Coritiba', 'fase': '3ª fase'}, {'clube': 'Remo', 'fase': '3ª fase'},
    # Ano 2024
    {'clube': 'Atlético-MG', 'fase': 'final'}, {'clube': 'Flamengo', 'fase': 'final'}, {'clube': 'Vasco da Gama', 'fase': 'semifinal'}, {'clube': 'Corinthians', 'fase': 'semifinal'}, {'clube': 'São Paulo', 'fase': 'quartas'}, {'clube': 'Bahia', 'fase': 'quartas'}, {'clube': 'Athletico-PR', 'fase': 'quartas'}, {'clube': 'Internacional', 'fase': 'oitavas'}, {'clube': 'Fluminense', 'fase': 'oitavas'}, {'clube': 'Palmeiras', 'fase': 'oitavas'}, {'clube': 'Botafogo', 'fase': 'oitavas'}, {'clube': 'Grêmio', 'fase': 'oitavas'}, {'clube': 'Ceará', 'fase': '3ª fase'}, {'clube': 'Fortaleza', 'fase': '3ª fase'}, {'clube': 'Sport', 'fase': '3ª fase'},
]



df = pd.DataFrame(dados_copa_brasil)
pontuacao_fase = {"1ª fase": 0, "2ª fase": 0.5, "3ª fase": 1, "oitavas": 2, "quartas": 3, "semifinal": 4, "final": 5, "campeão": 6}
df["fase"] = df["fase"].str.lower().str.strip()
df["pontuacao"] = df["fase"].map(pontuacao_fase)
pontuacoes = df.groupby("clube")["pontuacao"].sum().reset_index()
total_pontuacao = pontuacoes["pontuacao"].sum()
pontuacoes["probabilidade"] = pontuacoes["pontuacao"] / total_pontuacao

#Construção do grafo
grafo = GrafoManual()
for _, row in pontuacoes.iterrows():
    grafo.inserir_no(row['clube'])
for i in range(len(pontuacoes)):
    for j in range(i + 1, len(pontuacoes)):
        clube1, p1 = pontuacoes.iloc[i][["clube", "pontuacao"]]
        clube2, p2 = pontuacoes.iloc[j][["clube", "pontuacao"]]
        diff = abs(p1 - p2)
        if diff <= 4.0:
            grafo.inserir_aresta(clube1, clube2, diff)

# Executar o Kruskal 
mst_arestas = kruskal(grafo)
MST_nx = nx.Graph() # Criar grafo networkx para visualização
MST_nx.add_weighted_edges_from(mst_arestas) # arestas com peso

# Executar o Dijkstra
pontuacoes_sorted = pontuacoes.sort_values("pontuacao")
clube_inicio = pontuacoes_sorted.iloc[0]["clube"]
clube_fim = pontuacoes_sorted.iloc[-1]["clube"]
caminho, custo = dijkstra(grafo, clube_inicio, clube_fim)

# Impressão dos resultados
print("\n1. Análise com Algoritmo de Kruskal")
print(f"A Árvore Geradora Mínima (MST) calculada otimiza o grafo para {len(mst_arestas)} arestas.")
print("\n2. Análise com Algoritmo de Dijkstra")
print(f"\nCaminho mais suave de '{clube_inicio}' até '{clube_fim}':")
print(" -> ".join(caminho))
print(f"Custo total do caminho: {custo:.2f}")

# Plotar o grafico com os 5 melhores times
top_5_times = pontuacoes.sort_values("probabilidade", ascending=False).head(5)
fig, ax = plt.subplots(figsize=(12, 7))

cores = plt.cm.viridis(top_5_times['probabilidade'] / max(top_5_times['probabilidade']))
bars = ax.bar(top_5_times['clube'], top_5_times['probabilidade'], color=cores)
ax.set_title('Top 5 Clubes com Maior Probabilidade de Título\n(Baseado no Histórico 2021-2024)', fontsize=16)
ax.set_ylabel('Probabilidade (Pontuação / Total)', fontsize=12)
ax.set_xlabel('Clube', fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval:.2%}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()

# Plotar o grafo de rede
plt.figure(figsize=(22, 22))

pos = nx.spring_layout(MST_nx, k=1.5, iterations=100, seed=42)
pontuacoes_dict = pontuacoes.set_index('clube')['pontuacao']
node_sizes = [pontuacoes_dict.get(node, 0) * 200 + 150 for node in MST_nx.nodes()]
node_colors = [pontuacoes_dict.get(node, 0) for node in MST_nx.nodes()]

nx.draw_networkx_nodes(MST_nx, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.coolwarm, alpha=0.9)
nx.draw_networkx_edges(MST_nx, pos, alpha=0.7, edge_color="gray", width=1.5)
nx.draw_networkx_labels(MST_nx, pos, font_size=12, font_family="sans-serif", font_color="black")
plt.title("Grafo Otimizado com Kruskal - Copa do Brasil (Base de 20 Times)", size=24)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))),
             ax=plt.gca()).set_label('Pontuação Total Acumulada')
plt.axis('off')
plt.tight_layout()

plt.show()

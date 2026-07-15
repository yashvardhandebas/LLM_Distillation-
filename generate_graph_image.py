import networkx as nx
import matplotlib.pyplot as plt

graph_path = r"c:\Users\admin\Desktop\tsllm\apple_kg.graphml"
output_path = r"C:\Users\admin\.gemini\antigravity\brain\00d0c114-a476-46d6-875b-650a47c6ef32\apple_kg.png"

G = nx.read_graphml(graph_path)

plt.figure(figsize=(20, 14))

pos = nx.spring_layout(G, k=1.0, iterations=100)

nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=3500, alpha=0.9, edgecolors='black', linewidths=1.5)

nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='black')

nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=2.0, alpha=0.7)

edge_labels = nx.get_edge_attributes(G, 'type')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.5, font_color='darkred')

plt.title("Apple Inc. Knowledge Graph", fontsize=24, fontweight='bold', pad=20)
plt.axis('off')

plt.tight_layout()
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Graph image successfully saved to: {output_path}")

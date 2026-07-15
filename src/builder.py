import networkx as nx
import json

class KGBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_extraction(self, extraction_data):
        for ent in extraction_data.get('entities', []):
            self.graph.add_node(ent['name'], type=ent.get('type', 'Unknown'))
            
        for rel in extraction_data.get('relations', []):
            self.graph.add_edge(
                rel['source'], 
                rel['target'], 
                type=rel['type'], 
                confidence=rel.get('confidence', 1.0)
            )
        return self.graph

class KGSerializer:
    @staticmethod
    def to_json_ld(graph):
        nodes = [{"@id": n, **attr} for n, attr in graph.nodes(data=True)]
        links = [{"@source": u, "@target": v, **attr} for u, v, attr in graph.edges(data=True)]
        return json.dumps({"@graph": nodes, "links": links}, indent=2)

    @staticmethod
    def to_graphml(graph, filepath="graph.graphml"):
        nx.write_graphml(graph, filepath)
        return filepath

    @staticmethod
    def to_image(graph, title="Knowledge Graph", filepath="graph.png"):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 14))
        pos = nx.spring_layout(graph, k=1.0, iterations=100)
        
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=3500, alpha=0.9, edgecolors='black', linewidths=1.5)
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight='bold', font_color='black')
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True, arrowsize=20, width=2.0, alpha=0.7)
        
        edge_labels = nx.get_edge_attributes(graph, 'type')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, label_pos=0.5, font_color='darkred')
        
        plt.title(title, fontsize=24, fontweight='bold', pad=20)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

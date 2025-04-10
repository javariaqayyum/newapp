from flask import Flask, request, jsonify
import openai
import pandas as pd
from flask import render_template
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict  # Added missing import
import numpy as np

app = Flask(__name__)
# Important: Remove or properly secure your API key in production
openai.api_key = 'sk-proj-3HEcnxpNUAI8ZfplX8ihT3BlbkFJp0qmKBhRFjzB4u1RNQty'

# Initialize models
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge Graph Structure
class ProductKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.product_embeddings = {}
        
    def add_product(self, product):
        node_id = f"product_{product['id']}"
        self.graph.add_node(node_id, **product, type='product')
        
        # Add embeddings for RAG
        self.product_embeddings[node_id] = encoder.encode(
            f"{product['name']} {product['category']} {product['store']}"
        )
        
        # Connect to store
        store_node = f"store_{product['store']}"
        self.graph.add_node(store_node, type='store', name=product['store'])
        self.graph.add_edge(store_node, node_id, relationship='sells')
        
        # Connect to category
        category_node = f"category_{product['category']}"
        self.graph.add_node(category_node, type='category', name=product['category'])
        self.graph.add_edge(node_id, category_node, relationship='belongs_to')

# Initialize Knowledge Graph
kg = ProductKnowledgeGraph()

# Load data and build graph
def initialize_system():
    try:
        df = pd.read_csv('updated_file_data_final.csv')
        for idx, row in df.iterrows():
            try:
                product = {
                    'id': idx,
                    'name': row['Product Name'],
                    'price': float(row['Product Price'].replace('$', '').replace(',', '')),  # Added handling for commas
                    'store': row['Store Name'],
                    'category': row['Category'],
                    'link': row['Product Link']
                }
                kg.add_product(product)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
    except FileNotFoundError:
        print("Error: CSV file not found")
        exit(1)

initialize_system()

def rag_retrieval(query_embedding, threshold=0.7):
    """Retrieve relevant products using vector similarity"""
    similarities = {}
    for product_id, embedding in kg.product_embeddings.items():
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        if sim > threshold:
            similarities[product_id] = sim
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

def generate_cypher_query(natural_query):
    """Use OpenAI to convert natural language to Cypher query"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": """Convert this product search query into a Cypher query for our knowledge graph. 
                Consider synonyms and related categories. Respond only with the Cypher query."""
            }, {
                "role": "user",
                "content": natural_query
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating Cypher query: {e}")
        return None
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def enhanced_search():
    if not request.json or 'query' not in request.json:
        return jsonify({"error": "Missing query parameter"}), 400
    
    query = request.json['query']
    
    # Step 1: Semantic Search with RAG
    query_embedding = encoder.encode(query)
    rag_results = rag_retrieval(query_embedding)
    
    # Step 2: Knowledge Graph Query Expansion
    graph_results = []
    cypher_query = generate_cypher_query(query)
    if cypher_query:
        try:
            # Note: nx.cypher.read doesn't exist - you'll need to use a proper Neo4j driver
            # This is a placeholder for actual graph query execution
            graph_results = []  # Replace with actual graph query results
        except Exception as e:
            print(f"Graph query failed: {e}")
    
    # Step 3: Combine results
    combined_results = combine_results(rag_results, graph_results)
    
    # Step 4: Store aggregation and pricing
    store_totals = defaultdict(lambda: {'items': [], 'total': 0.0})
    for product in combined_results:
        store = product['store']
        store_totals[store]['items'].append(product)
        store_totals[store]['total'] += product['price']
    
    # Step 5: Sorting logic
    sorted_stores = sorted(
        store_totals.items(),
        key=lambda x: (-len(x[1]['items']), x[1]['total'])
    )
    
    # Format response
    response = []
    for store, data in sorted_stores:
        response.append({
            'store': store,
            'total': round(data['total'], 2),
            'item_count': len(data['items']),
            'items': sorted(data['items'], key=lambda x: x['price'])
        })
    
    return jsonify(response)

def combine_results(rag_results, graph_results):
    # Implement hybrid scoring logic
    combined = {}
    
    # Add RAG results
    for product_id, score in rag_results:
        node = kg.graph.nodes[product_id]
        combined[product_id] = {
            **node,
            'score': score * 0.7  # Weight for RAG results
        }
    
    # Add Graph results
    for record in graph_results:
        # Note: This assumes record format needs to be adjusted based on actual graph query results
        product_id = record.get('id', None)  # Adjust based on your actual graph query results
        if product_id and product_id in kg.graph.nodes:
            node = kg.graph.nodes[product_id]
            if product_id in combined:
                combined[product_id]['score'] += 0.3  # Weight for graph results
            else:
                combined[product_id] = {
                    **node,
                    'score': 0.3
                }
    
    return sorted(combined.values(), key=lambda x: x['score'], reverse=True)

if __name__ == '__main__':
    app.run(debug=True)
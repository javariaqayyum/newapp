from flask import Flask, request, render_template, jsonify
import pandas as pd
from collections import defaultdict
from collections import Counter
import openai
import json

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/newinsta'

# Configure OpenAI API
openai.api_key = 'sk-proj-3HEcnxpNUAI8ZfplX8ihT3BlbkFJp0qmKBhRFjzB4u1RNQty'

def load_dataset():
    df = pd.read_csv('updated_file_data_final.csv')
    required_columns = ['Product Name', 'Product Price', 'Store Name', 'Category', 'Product Link']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df.to_dict('records')

products = load_dataset()

import re

def extract_suggestions(gpt_output):
    # Split numbered list (1. Item 2. Item etc.) OR dash list
    lines = re.split(r'\d+\.\s*|\n+|- ', gpt_output)
    # Clean and filter empty strings
    return [line.strip() for line in lines if line.strip()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({'error': 'Invalid request format'}), 400
    
    items = data['items']
    products = load_dataset()
    all_results = []
    
    for query in items:
        keyword_list = [query.strip().lower()]
        filtered_products = []
        for product in products:
            product_name = product.get('Product Name', '').lower()
            raw_price = product.get('Product Price', '0')
            try:
                price = float(raw_price.replace('$', '').strip())
            except (ValueError, AttributeError):
                price = 0
            
            if any(keyword in product_name for keyword in keyword_list) and price > 0:
                filtered_products.append({
                    "Product Name": product.get('Product Name'),
                    "Product Price": price,
                    "Site": product.get('Store Name'),
                    "Product Link": product.get('Product Link')
                })

        keyword_store_items = defaultdict(lambda: defaultdict(list))
        for product in filtered_products:
            product_name = product['Product Name'].lower()
            matched_keywords = [kw for kw in keyword_list if kw in product_name]
            for kw in matched_keywords:
                keyword_store_items[kw][product['Site']].append(product)

        store_data = defaultdict(dict)
        for kw in keyword_list:
            stores_for_kw = keyword_store_items.get(kw, {})
            for store, store_products in stores_for_kw.items():
                non_zero_products = [p for p in store_products if p['Product Price'] > 0]
                if non_zero_products:
                    cheapest = min(non_zero_products, key=lambda x: x['Product Price'])
                    store_data[store][kw] = {
                        'Product Name': cheapest['Product Name'],
                        'Product Price': cheapest['Product Price'],
                        'Product Link': cheapest['Product Link']
                    }

        for store, items in store_data.items():
            store_entry = {
                'Store': store,
                'Items': [],
                'Total': 0.0,
                'OriginalQuery': query
            }
            for kw, details in items.items():
                store_entry['Items'].append({
                    'Item': kw,
                    'ProductName': details['Product Name'],
                    'Price': details['Product Price'],
                    'Link': details['Product Link']
                })
                store_entry['Total'] += details['Product Price']
            store_entry['Total'] = round(store_entry['Total'], 2)
            all_results.append(store_entry)
    
    final_results = defaultdict(lambda: {'Store': '', 'Items': [], 'Total': 0.0})
    for result in all_results:
        store = result['Store']
        final_results[store]['Store'] = store
        final_results[store]['Items'].extend(result['Items'])
        final_results[store]['Total'] += result['Total']
    
    formatted_result = list(final_results.values())
    formatted_result = sorted(formatted_result, key=lambda x: (-len(x['Items']), x['Total']))
    return jsonify(formatted_result)

@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    data = request.get_json()
    if not data or 'items' not in data:
        return jsonify({'error': 'Invalid request format'}), 400

    # Load the product names from the dataset
    df = pd.read_csv('updated_file_data_final.csv')
    product_names = df['Product Name'].dropna().astype(str).str.lower().tolist()

    suggestion_results = {}

    # Loop through each search term provided by the user
    for query in data['items']:
        query = query.strip().lower()  # Clean the input and convert to lowercase
        
        if not query:
            continue

        # Ensure that we are adding a space at the end of the query to match exact phrases
        query_with_space = query + ' '  # Add a space at the end
        
        # Get all product names that contain the exact query with space
        matches = [name for name in product_names if query_with_space in name]

        phrases = []
        for name in matches:
            tokens = name.split()
            for i, token in enumerate(tokens):
                if query in token:
                    start = max(0, i - 2)
                    end = min(len(tokens), i + 3)
                    phrase = " ".join(tokens[start:end])
                    phrases.append(phrase.title())

        # Get the top N most common phrases (you can increase this number to get more suggestions)
        top_phrases = [phrase for phrase, _ in Counter(phrases).most_common(15)]  # Increase the number here for more options

        # Optionally refine the suggestions using OpenAI if needed
        refined_suggestions = refine_suggestions_with_openai(query, top_phrases)
        suggestion_results[query] = refined_suggestions

    return jsonify(suggestion_results)

import re


def extract_suggestions(response_text):
    """
    Extracts suggestions from GPT-4 response formatted as numbered list or line-by-line.
    Handles '1. Item name', '• Item name', or plain lines.
    """
    lines = response_text.strip().split('\n')
    cleaned = []

    for line in lines:
        # Remove numbering like "1. ", "2) ", or bullet "• "
        cleaned_line = re.sub(r'^\s*(\d+[\.\)]|\•)\s*', '', line)
        if cleaned_line:
            cleaned.append(cleaned_line.strip())

    return cleaned

def refine_suggestions_with_openai(query, suggestions):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Format the following product suggestions into a clean numbered list (1., 2., 3., etc). 
Do not summarize, merge, or remove any items. Keep every item in the list exactly as it is. 
Only format for clarity and consistency.

            Suggestions: {', '.join(suggestions)}"""}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=400,
            temperature=0.2
        )

        refined_text = response['choices'][0]['message']['content'].strip()
        refined_suggestions = extract_suggestions(refined_text)

        # Extract the refined suggestions from the response
        refined_text = response['choices'][0]['message']['content']
        refined_suggestions = extract_suggestions(refined_text)

        return refined_suggestions
    except Exception as e:
        print(f"Error refining suggestions with GPT-4: {e}")
        return suggestions  # Return the original suggestions if there's an error
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

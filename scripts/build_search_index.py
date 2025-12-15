import os
import json
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove extra whitespace and newlines
    return ' '.join(text.split())

def build_search_index(root_dir):
    index = []
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip assets and src directories
        if 'assets' in dirpath or 'src' in dirpath or '.git' in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith('.html'):
                file_path = os.path.join(dirpath, filename)
                
                # Calculate relative URL from root
                rel_path = os.path.relpath(file_path, root_dir)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                soup = BeautifulSoup(content, 'html.parser')
                
                # Get title
                title_tag = soup.find('title')
                title = title_tag.text.split('|')[0].strip() if title_tag else filename
                
                # Get main content
                main_content = soup.find('main') or soup.find('body')
                if main_content:
                    # Remove script and style elements
                    for script in main_content(["script", "style"]):
                        script.decompose()
                    
                    text_content = clean_text(main_content.get_text())
                    
                    # Add to index
                    index.append({
                        'title': title,
                        'url': rel_path,
                        'content': text_content[:5000] # Limit content size for performance
                    })
                    print(f"Indexed: {rel_path}")

    # Save to assets/js/search_index.json
    output_path = os.path.join(root_dir, 'assets', 'js', 'search_index.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False)
        
    print(f"\nSearch index built with {len(index)} pages.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    # Assume script is run from root or proper location
    if current_dir.endswith('scripts'):
        root = os.path.dirname(current_dir)
    else:
        root = current_dir
        
    build_search_index(root)

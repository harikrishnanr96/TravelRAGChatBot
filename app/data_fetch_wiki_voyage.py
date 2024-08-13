import requests
from bs4 import BeautifulSoup
import pandas as pd
import string

url = 'https://en.wikivoyage.org/wiki/London'

response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.content, 'html.parser')

# Find the content division of the page
content_div = soup.find(id='mw-content-text')

# Initialize a list to hold the data
data = []

# Function to clean text
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Variables to hold the current section and subsection titles
section_title = ''
subsection_title = ''
section_text = []

# Iterate through sections in the content
for section in content_div.find_all(['h2', 'h3', 'p']):
    if section.name == 'h2':
        if section_text:
            data.append({'Section': section_title, 'Subsection': subsection_title, 'Content': ' '.join(section_text)})
            section_text = []
        section_title = section.get_text(strip=True)
        subsection_title = ''
    elif section.name == 'h3':
        if section_text:
            data.append({'Section': section_title, 'Subsection': subsection_title, 'Content': ' '.join(section_text)})
            section_text = []
        subsection_title = section.get_text(strip=True)
    elif section.name == 'p':
        # Clean the text inside paragraph tags
        cleaned_text = clean_text(' '.join(section.get_text(strip=True).split()))
        section_text.append(cleaned_text)

# # Save the last section
if section_text:
    data.append({'Section': section_title, 'Subsection': subsection_title, 'Content': ' '.join(section_text)})

# Create a DataFrame from the data
df = pd.DataFrame(data)

df.to_csv('london_travel_data.csv', index=False)


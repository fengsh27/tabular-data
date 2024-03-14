import requests  
  
pmc_id = "PMC1721266"  # Replace with your PMC ID  
url = f"https://www.ncbi.nlm.nih.gov/research/pmcapi/articles/resolve/{pmc_id}?format=txt"  
  
response = requests.get(url)  
  
if response.status_code == 200:  
    with open(f'{pmc_id}.txt', 'w') as f:  
        f.write(response.text)  
else:  
    print(f"Unable to download article with PMC ID {pmc_id}. Status code: {response.status_code}")  

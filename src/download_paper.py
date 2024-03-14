from Bio import Entrez  
  
# Always tell NCBI who you are  
Entrez.email = "shaohong.feng@osumc.edu"  
  
def fetch_article(pmid):  
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml', rettype='full')  
    return Entrez.read(handle)  
  
def print_article(article):  
    print(article['MedlineCitation']['Article']['ArticleTitle'])  
  
# Fetch article  
article = fetch_article('11420324')  
  
# Print title  
print_article(article)  

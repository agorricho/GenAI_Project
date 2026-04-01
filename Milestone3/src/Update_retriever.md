Update(GenAI_Project/Milestone3/src/agents/retriever.py)                                ──────────────────── fix-pipeline-auth-wire-chat ──
  ⎿  Added 10 lines, removed 21 lines                                                                                                        
      11  """                                                                                                                                
      12                                                                                                                                     
      13  import os                                                                                                                          
      14 -import requests                                                                                                                    
      14  from pathlib import Path                                                                                                           
      15  from dotenv import load_dotenv                                                                                                     
      16 +from fastembed import TextEmbedding                                                                                                
      17  from qdrant_client import QdrantClient                                                                                             
      18                                                                                                                                     
      19  # ── Load credentials ──────────────────────────────────────────────────────────
     ...
      22          load_dotenv(_p / ".env")
      23          break
      24  
      25 -QDRANT_URL      = os.getenv("QDRANT_URL")                                                                                   
      26 -QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")                                                                               
      27 -OLLAMA_API_KEY  = os.getenv("0LLAMA", "")                                                                                   
      28 -OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:11434") + "/api/embeddings"                         
      29 -EMBED_MODEL     = "nomic-embed-text"                                                                                        
      30 -COLLECTION      = "msa8700_m2"                                                                                              
      31 -TOP_K           = 5                                                                                                         
      32 -TEXT_LIMIT      = 1500                                                                                                      
      25 +QDRANT_URL     = os.getenv("QDRANT_URL")                                                                                    
      26 +QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")                                                                                
      27 +COLLECTION     = "msa8700_m2"                                                                                               
      28 +TOP_K          = 5                                                                                                          
      29  
      30  # ── Qdrant client ─────────────────────────────────────────────────────────────
      31  _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
      32  
      33 +# ── Embedding model (ONNX, compatible with nomic-embed-text vectors in Qdrant) ─                                           
      34 +_embedder = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")                                                                 
      35  
      36 +                                                                                                                            
      37  def _embed(text: str) -> list[float]:
      39 -    """Embed text via Ollama (mirrors M2 embed_with_retry logic)."""                                                        
      40 -    headers = {}                                                                                                            
      41 -    if OLLAMA_API_KEY:                                                                                                      
      42 -        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"                                                               
      43 -    resp = requests.post(                                                                                                   
      44 -        OLLAMA_EMBED_URL,                                                                                                   
      45 -        headers=headers,                                                                                                    
      46 -        json={"model": EMBED_MODEL, "prompt": text[:TEXT_LIMIT], "keep_alive": "5m"},                                       
      47 -        timeout=60,                                                                                                         
      48 -    )                                                                                                                       
      49 -    resp.raise_for_status()                                                                                                 
      50 -    return resp.json()["embedding"]                                                                                         
      38 +    """Embed text via fastembed (nomic-embed-text, ONNX — same vectors as Ollama)."""                                       
      39 +    return list(next(iter(_embedder.embed([text]))))                                                                        
      40  
      41  
      42  # ── Node function ─────────────────────────────────────────────────────────────
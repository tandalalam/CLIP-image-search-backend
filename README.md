# CLIP Image Search Backend  

A backend for a search engine that uses the **CLIP model** and **Qdrant vector database** to match text queries with relevant images. Ideal for product discovery and cross-modal search applications.  

---

## Features  
- Converts text queries into image-space embeddings.  
- Retrieves images that best match the textual description.  
- Built using **CLIP** and **Qdrant** for high performance.  

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/tandalalam/CLIP-image-search-backend.git  
   cd CLIP-image-search-backend/src  
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  
   ```  

3. Set up a Qdrant instance and configure the `.env` file:  
   ```env
   QDRANT_HOST=<your_qdrant_host>:<port>  
   ```  

4. Run the server:  
   ```bash
   python main.py  
   ```  

You can also run the project using Docker by simply `docker build -t clip-search` and then `docker run -p 8080:8080 clip-search`.

---

## Usage  

Send a POST request with a text query to retrieve matching images:  
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "red shoes"}' http://localhost:8080/search  
```  

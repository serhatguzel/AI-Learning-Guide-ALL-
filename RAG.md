# 🧠 RAG (Retrieval-Augmented Generation) Mimarisi

Bu belge, Büyük Dil Modellerinin (LLM) harici ve özelleştirilmiş veri tabanlarıyla entegre çalışmasını sağlayan **RAG (Geri Çağırımla Artırılmış Üretim)** mimarisinin temel kavramlarını, bileşenlerini ve genel akışını açıklar.

![RAG Mimarisi](https://img.shields.io/badge/Architecture-RAG-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LLM](https://img.shields.io/badge/LLM-HuggingFace%20%7C%20Gemini-orange?style=for-the-badge)

## 📌 İçindekiler
- [Nedir?](#-nedir)
- [Neden RAG?](#-neden-rag)
- [Mimari Bileşenleri](#-mimari-bileşenleri)
- [Sistem Akışı](#-sistem-akışı)
- [Örnek Teknoloji Yığını](#-örnek-teknoloji-yığını)
- [Temel Kullanım Örneği (Python)](#-temel-kullanım-örneği-python)

---

## 📖 Nedir?
**RAG**, standart bir dil modelinin (LLM) halüsinasyon (yanlış bilgi üretme) riskini en aza indirmek için, kullanıcıya yanıt üretmeden önce güvenilir bir "dış bilgi tabanından" (örneğin kurum içi dokümanlar, veritabanları) ilgili bilgileri arayıp getirmesi ve bu bilgileri bağlam olarak kullanarak yanıt oluşturması sürecidir.

## 🚀 Neden RAG?
- **Güncellik:** Modelin baştan eğitilmesine (Fine-tuning) gerek kalmadan yeni verilere anında erişim sağlar.
- **Doğruluk ve Kaynak Gösterme:** Üretilen yanıtların hangi belgeye/veriye dayandığı net olarak izlenebilir.
- **Veri Gizliliği:** Özel veya hassas kurumsal veriler genel LLM eğitimine dahil edilmeden, izole bir vektör veritabanında tutulur.

## 🏗️ Mimari Bileşenleri

1. **Document Loader (Belge Yükleyici):** PDF, HTML, Markdown, TXT veya veritabanı kayıtlarının sisteme alınması.
2. **Text Splitter (Metin Bölütleyici - Chunking):** Uzun metinlerin, LLM'in bağlam penceresine (context window) sığacak anlamlı ve küçük parçalara bölünmesi.
3. **Embedding Model (Gömme Modeli):** Metin parçalarını çok boyutlu uzayda sayısal vektörlere dönüştüren model (örn. Hugging Face modelleri, OpenAI Embeddings).
4. **Vector Database (Vektör Veritabanı):** Vektörleştirilmiş verilerin saklandığı ve hızlı semantik (anlamsal) aramanın yapıldığı veritabanı.
5. **Retriever (Geri Çağırıcı):** Kullanıcı sorgusunu vektöre çevirip, veritabanındaki en benzer "K" (top-k) parçayı getiren mekanizma.
6. **Generator (Üretici - LLM):** Getirilen bağlamı ve orijinal kullanıcı sorgusunu harmanlayarak nihai yanıtı üreten büyük dil modeli.

## 🔄 Sistem Akışı

Sistem iki ana fazdan oluşur:

### 1. Çevrimdışı Faz (Veri Hazırlama / İndeksleme)
`Ham Veri` ➔ `Metin Parçalama (Chunking)` ➔ `Vektörleştirme (Embedding)` ➔ `Vektör Veritabanına Kayıt`

### 2. Çevrimiçi Faz (Sorgu ve Üretim)
`Kullanıcı Sorgusu` ➔ `Sorgunun Vektörleştirilmesi` ➔ `Vektör DB'de Semantik Arama` ➔ `Alakalı Parçaların (Context) Bulunması` ➔ `Sorgu + Context = Genişletilmiş Prompt` ➔ `LLM` ➔ `Nihai Yanıt`

## 🛠️ Örnek Teknoloji Yığını

Yüksek performanslı bir backend ve AI entegrasyonu için tavsiye edilen stack:

- **Orkestrasyon:** LangChain, LlamaIndex
- **Dil:** Python
- **Embedding Modelleri:** Hugging Face (`sentence-transformers`), Google Generative AI Embeddings
- **Vektör Veritabanı:** ChromaDB, Qdrant, Milvus veya pgvector (PostgreSQL)
- **LLM:** Google Gemini, Claude, Llama 3 (Lokal kullanım için Ollama)
- **API Katmanı:** FastAPI (Mikroservis yapısına uygun uç noktalar için)

## 💻 Temel Kullanım Örneği (Python)

Aşağıda temel bir RAG akışının LangChain ve Chroma kullanılarak nasıl kurgulanabileceğine dair sözde kod (pseudo-code) örneği bulunmaktadır:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Veriyi Yükle ve Parçala
loader = TextLoader("data/kurumsal_bilgiler.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Embedding Modeli ve Vektör Veritabanı
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. Retriever (Geri Çağırıcı) Oluştur
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. LLM ve Prompt Tanımlama
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = PromptTemplate.from_template(
    "Aşağıdaki bağlamı kullanarak soruyu yanıtla:\nBağlam: {context}\nSoru: {question}"
)

# 5. Sorguyu Çalıştır (RAG Zinciri)
soru = "Projenin temel amacı nedir?"
ilgili_belgeler = retriever.invoke(soru)
baglam = "\n".join([doc.page_content for doc in ilgili_belgeler])

cevap = llm.invoke(prompt.format(context=baglam, question=soru))
print(cevap.content)
```

---
*Bu yapı, karmaşık doküman analizi, müşteri destek asistanları ve kurumsal veri tabanlarında semantik arama projeleri için temel bir iskelet oluşturur.*

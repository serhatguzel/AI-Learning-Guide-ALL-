# LlamaIndex ve LangChain Karşılaştırması

Yapay zeka ekosisteminde Büyük Dil Modelleri (LLM) ile uygulama geliştirmek için kullanılan en popüler iki framework olan **LlamaIndex** ve **LangChain**'in detaylı karşılaştırması, kullanım senaryoları ve kod örnekleri.

---

## 1. LlamaIndex (Veri Odaklı Uzman)

LlamaIndex'in temel varoluş amacı, özel verilerinizi (PDF, Notion, SQL vb.) LLM'lere en verimli şekilde yedirmektir. RAG (Retrieval-Augmented Generation) mimarisinin kalbini oluşturur.

### Avantajları (Pros)
* **Hız ve Kolaylık:** Gelişmiş bir RAG sistemini (veriyi alma, bölme, indeksleme, arama) kelimenin tam anlamıyla 5-10 satır kodla ayağa kaldırabilirsin.
* **Veri Bağlayıcı Zenginliği (Data Ingestion):** LlamaHub aracılığıyla Slack, Google Drive, Jira, GitHub gibi yüzlerce farklı kaynaktan veriyi tek tıkla çekebilir.
* **Gelişmiş İndeksleme Yöntemleri:** Sadece vektör (anlam) araması yapmaz; verileri ağaç yapısı (Tree Index), anahtar kelime tablosu veya SQL yapıları halinde dizebilir. Bu da karmaşık belgelerde çok daha isabetli sonuçlar getirir.
* **Optimize Edilmiş Arama (Retrieval):** LlamaIndex, sorulan soruyu veritabanında aratırken arka planda soruyu parçalama, alt sorular üretme (Sub-Question Query Engine) veya sonuçları yeniden sıralama (Re-ranking) gibi ileri düzey teknikleri otomatik veya çok kolay bir şekilde sunar.

### Dezavantajları (Cons)
* **Ajan (Agent) Yeteneklerinde Sınırlılık:** Son güncellemelerle kendi ajan yapılarını (Agentic RAG) kursa da, karmaşık mantık zincirleri oluşturmak ve farklı araçları (API'ler, hesap makineleri) kullanmak konusunda LangChain kadar esnek ve köklü değildir.
* **Dar Odak:** Eğer uygulaman sadece belge okumak üzerine değilse (örneğin internette sörf yapıp fiyat karşılaştıran bir bot yazacaksan), LlamaIndex fazla dar bir araç kalabilir.

---

## 2. LangChain (Orkestrasyon ve Aksiyon Uzmanı)

LangChain, bir LLM'in sadece metin üretmesini değil, bir "beyin" gibi hareket edip dış dünyayla etkileşime girmesini (Ajanlar/Agents) sağlamak için tasarlanmıştır.

### Avantajları (Pros)
* **Devasa Esneklik:** Neredeyse piyasadaki her LLM modelini, her vektör veritabanını ve yüzlerce farklı aracı birbiriyle lego gibi birleştirebilirsin.
* **Güçlü Ajan (Agent) Yapıları:** Bir ajana "İşte bir hesap makinesi, işte Google arama motoru, işte bir SQL veritabanı. Kullanıcının sorusunu çözmek için bunları kendi mantığınla kullan" diyebilirsin (Tool Use / Function Calling). LangChain bu süreci çok iyi yönetir.
* **Zincirleme Mantığı (Chains):** Karmaşık görevleri alt görevlere bölmek. Örneğin: "Önce metni özetle -> Sonra özetin içinden isimleri çıkar -> Sonra bu isimleri bir API'ye gönder."
* **Topluluk ve Ekosistem:** Çok daha büyük bir topluluğa sahiptir. Neredeyse her kullanım senaryosu için hazır bir LangChain entegrasyonu (LangSmith, LangServe) bulmak mümkündür.

### Dezavantajları (Cons)
* **Dik Öğrenme Eğrisi:** LlamaIndex'te 5 satırda yapılan bir RAG işlemi, LangChain'de çok fazla manuel yapılandırma, doküman yükleyici, metin bölücü, bellek yönetimi (Memory) ayarı gerektirebilir.
* **Gereksiz Karmaşıklık (Overengineering):** Eğer tek yapmak istediğin bir PDF'e soru sormaksa, LangChain kullanmak bazen sinek öldürmek için balyoz kullanmaya benzer. Basit işler için fazla karmaşık bir yapısı (çok fazla "wrapper" ve soyutlama) vardır.
* **Gelişmiş Arama Eksikliği:** Temel RAG işlemlerini çok iyi yapsa da, LlamaIndex'in sunduğu ileri düzey indeksleme ve veriyi yapılandırma (Data Structuring) araçları LangChain'de o kadar zarif veya entegre değildir.

---

## Hangisini Ne Zaman Seçmeli?

| Senaryo | LlamaIndex | LangChain |
| :--- | :--- | :--- |
| **Şirket içi PDF/Doküman yığınlarına soru sorma (Temel RAG)** | 🥇 İlk Tercih (Çok daha kolay ve temiz) | 🥈 Yapılabilir ama daha zahmetli |
| **İnternette arama yapabilen, API çağırabilen sanal asistan** | 🥈 Gelişiyor ama sınırlı | 🥇 İlk Tercih (Agentic Workflow) |
| **Karmaşık ve yapısal olmayan veri kaynaklarını işleme** | 🥇 Mükemmel Indexleme algoritmaları | 🥈 Sadece temel Vektör Store kullanımı |
| **"Önce bunu yap, sonucu şuraya gönder" gibi zincirleme işler** | 🥈 Odak noktası bu değil | 🥇 Bu iş için tasarlandı (Chains) |

> **💡 Endüstri Standardı:**
> Piyasada, özellikle kurumsal üretim (production) projelerinde, LlamaIndex ve LangChain çoğu zaman **birlikte kullanılır**. LlamaIndex veriyi toplamak, indekslemek ve en iyi parçayı bulmak için (Retrieval Engine) kullanılırken; LangChain bu motoru çağıran, hafızayı tutan ve diğer API'lerle etkileşime giren ana beyni (Agent) oluşturmak için kullanılır.

---

## Kod Örnekleri: Aynı İşlem İçin İki Yaklaşım

**Senaryo:** `sirket_politikasi.txt` dosyasını okutup *"İzin günleri nasıl hesaplanır?"* diye sormak. İki framework arasındaki felsefe ve kod yapısı farkını aşağıda görebilirsiniz.

### 1. LlamaIndex ile (Kısa, Odaklı, Sihirli)
LlamaIndex arka planda parçalama (chunking), embedding (vektöre çevirme) ve arama işlerini senin yerine varsayılan ayarlarla yapar. Sadece "veriyi al ve soruyu sor" dersin.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. Klasördeki tüm belgeleri (PDF, TXT) otomatik oku
documents = SimpleDirectoryReader('veriler/').load_data()

# 2. Veriyi indeksle (Arka planda parçalar ve vektör veritabanına atar)
index = VectorStoreIndex.from_documents(documents)

# 3. İndeksi bir sorgu motoruna çevir
query_engine = index.as_query_engine()

# 4. Soruyu sor
response = query_engine.query("Şirket politikasında izin günleri nasıl hesaplanır?")
print(response)
```
> **Felsefesi:** *Bana veriyi ver, arka plandaki karmaşık işleri ben en iyi standartlarda (best practices) hallederim.*

### 2. LangChain ile (Manuel, Zincirleme, Esnek)
LangChain, legoları birleştirmeni ister. Okuyucuyu sen seçersin, metni nasıl böleceğini sen söylersin, veritabanını sen eklersin, LLM ile veritabanını bağlayacak "zinciri" sen kurarsın.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Belgeyi yükle
loader = TextLoader("veriler/sirket_politikasi.txt")
docs = loader.load()

# 2. Metni belirli boyutlarda parçalara böl (Chunking)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Embedding modelini ve Vektör Veritabanını ayarla
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever() # Arama motoru yap

# 4. Bir LLM ve Sistem Prompt'u (Talimatı) tanımla
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("""
Aşağıdaki içeriğe göre soruyu cevapla.
İçerik: {context}
Soru: {input}
""")

# 5. Zincirleri kur (Retrieval Chain)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Soruyu sor
response = rag_chain.invoke({"input": "Şirket politikasında izin günleri nasıl hesaplanır?"})
print(response["answer"])
```
> **Felsefesi:** *Sana her adımın kontrolünü veriyorum. İstersen belgenin sadece 500 kelimesini al, istersen promptu tamamen değiştir, istersen bu aramanın sonucunu alıp başka bir API'ye gönder.*

---

## Özet

| Özellik | LlamaIndex (Veri Odaklı) | LangChain (Aksiyon Odaklı) |
| :--- | :--- | :--- |
| **Kurulum ve Kod Uzunluğu** | Çok kısa, 5-6 satırda hazır yapı | Daha uzun, manuel zincirleme gerektirir |
| **Belge Okuma (PDF vb.)** | Yerleşik ve çok güçlü | Dış kütüphanelere bağımlı |
| **İndeksleme Yeteneği** | Gelişmiş (Tree, Keyword, SQL vb.) | Temel seviye (Genelde vektör db) |
| **Dış Araç Kullanımı (Agents)** | Sınırlı, yeni yeni gelişiyor | Çok güçlü, sınırsız esneklik |
| **Öğrenme Eğrisi** | Basit (Veri odaklı işler için) | Dik (Çok fazla konsept var) |

**Sonuç:** Sıfırdan bir RAG mimarisi inşa edip verilerle konuşmak hedefleniyorsa, **LlamaIndex** ile başlamak arka plandaki matematiği ve vektör mantığını anlamak açısından çok daha hızlı ve tatmin edici bir başlangıç olacaktır.

# ChatGPT Prompt Mühendisliği: Çıkarım (Inferring) Teknikleri Çalışma Raporu

## 1. Giriş: Modern NLP ve Çıkarım Süreçlerinde Paradigma Değişimi

Doğal Dil İşleme (NLP) dünyasında, bir metinden anlam çıkarma, etiketleme ve veri ayrıştırma süreçleri köklü bir paradigma değişimi yaşamaktadır. Geleneksel Makine Öğrenmesi (ML) iş akışlarında, duygu analizi veya varlık çıkarma (NER) gibi görevler için BERT veya RoBERTa gibi modellerin özel olarak eğitilmesi, etiketli veri setlerinin toplanması ve her bir görev için ayrı bir modelin bulutta konuşlandırılması gerekiyordu.

Büyük Dil Modelleri (LLM) ile bu süreç, yerini tek bir API çağrısı ve stratejik prompt tasarımına bırakmıştır. Bu değişim, kurumsal projelerde "pazara sürüm süresi" (speed-to-market) ve ROI üzerinde devrimsel bir etki yaratmaktadır. Artık haftalar süren model eğitim döngüleri yerine, dakikalar içinde üretim kalitesinde (production-ready) çıkarım sistemleri kurulabilmektedir.

**Geleneksel ML vs. LLM Tabanlı Çıkarım Karşılaştırması**

| Özellik | Geleneksel ML (BERT/RoBERTa vb.) | Büyük Dil Modelleri (LLM) |
| :--- | :--- | :--- |
| **Geliştirme Hızı** | Haftalar veya aylar süren eğitim/test. | Dakikalar içinde prompt prototipleme. |
| **Model Yönetimi** | Her görev için ayrı model (Silo yapı). | Tek bir model/API ile sınırsız görev. |
| **Veri Gereksinimi** | Binlerce etiketli örnek (Supervised). | Sıfır veya çok az örnek (Zero/Few-shot). |
| **Altyapı** | Karmaşık GPU kümeleri ve deployment. | Tek bir API uç noktası üzerinden servis. |

Bu hız ve esneklik, tek bir modelin aynı anda hem duygu analizi yapıp hem de isim çıkarabilmesi sayesinde yazılım mimarilerini sadeleştirmekte ve operasyonel maliyetleri düşürmektedir.

## 2. Duygu Analizi (Sentiment Analysis): Veriyi Aksiyona Dönüştürme

Müşteri geri bildirimlerini anlamak, bir işletmenin stratejik yol haritasını belirleyen en kritik unsurdur. Duygu analizi, yapılandırılmamış müşteri sesini, iş zekası sistemleri için anlamlı birer veri noktasına dönüştürür.

**Vaka Analizi (Lumina Lamba):** Kaynak metinde belirtilen "Lumina Lamba" incelemesini ele alalım. Geleneksel yöntemlerin aksine, LLM'ler bağlamı (ek depolama alanı gibi nüansları) hızla kavrar.

**Post-Processing Dostu Prompt Tasarımı:** Yazılımsal otomasyonlarda modelin uzun cümleler kurması "kırılgan" (brittle) bir yapı oluşturur. Bu nedenle çıktıyı programatik olarak işlenebilecek tek bir kelimeye indirgemek esastır.

    import openai

    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
        return response.choices[0].message["content"]

    review = "Needed a nice lamp for the bedroom and this one had additional storage..."

    # Delimiters (''') kullanımı enjeksiyonu önler ve odaklanmayı artırır
    prompt = f'''
    What is the sentiment of the following product review, 
    which is delimited with triple backticks?

    Give your answer as a single word, either "positive" or "negative".

    Review text: ```{review}```
    '''

    response = get_completion(prompt)
    print(response) # Çıktı: "positive"

## 3. Hedefli Duygu ve Öfke Tespiti: Müşteri İlişkileri Yönetimi

Müşteri Başarı (Customer Success) ekipleri için en kritik veri, müşterinin sadece mutsuz olması değil, "öfkeli" olmasıdır. Öfke tespiti, bir destek biletinin (ticket) aciliyetini belirlemek için en güçlü alarm mekanizmasıdır.

**Öfke Tespiti Prompt Tasarımı:** Modelden ikili (binary) bir karar vermesini isteyerek sistemimizi otomatize edebiliriz:
*"Is the writer of the following review expressing anger? Answer with either 'yes' or 'no'."*

Bu tür bir "erken uyarı sistemi", öfkeli müşterilere anında geri dönüş yapılmasını sağlayarak müşteri kaybını (churn) minimize eder. Geleneksel sistemlerde bu seviyede bir duyarlılık için çok ciddi bir veri etiketleme süreci gerekirdi.

## 4. Zengin Bilgi Çıkarımı ve Yapılandırılmış Veri (JSON) Entegrasyonu

E-ticaret analitiğinde ham metinden ürün ve marka bilgilerini çekmek (Information Extraction), veri madenciliğinin kalbidir. LLM'lerin en büyük gücü, bu bilgileri doğrudan yazılım sistemlerinin konuşabileceği JSON formatında sunabilmesidir.

**JSON Formatlama Stratejisi:** JSON kullanımı, çıktının Python sözlüklerine (dictionary) doğrudan `json.loads()` ile aktarılabilmesini sağlar ve manuel ayrıştırma (parsing) hatalarını engeller.

    import json

    prompt = f'''
    Identify the following items from the review text: 
    - Item purchased
    - Company that made the item

    The review is delimited with triple backticks. 
    Format your response as a JSON object with "Item" and "Brand" as the keys. 
    If the information isn't present, use "unknown" as the value.
    Make your response as short as possible.

    Review text: ```{review}```
    '''

    response = get_completion(prompt)
    data = json.loads(response)
    print(data["Brand"]) # Çıktı: Lumina

## 5. Tek Seferde Çoklu Çıkarım (Joint Extraction) Teknikleri

Maliyet optimizasyonu ve düşük gecikme süresi (latency) için, birden fazla çıkarım görevini tek bir API çağrısında birleştirmek profesyonel bir yaklaşımdır. Ayrı ayrı API çağrıları yapmak yerine "Joint Extraction" tekniği kullanılır.

**Veri Tipi Manipülasyonu:** Öfke durumunun JSON içinde bir "boolean" (true/false) olarak istenmesi, kod tarafındaki mantıksal sorguları (`if not anger:`) daha sağlam hale getirir.

    prompt = f'''
    Identify the following items from the review text: 
    1. Sentiment (positive or negative)
    2. Is the reviewer expressing anger? (true or false)
    3. Item purchased
    4. Company that made it

    Format your response as a JSON object with \
    "sentiment", "anger", "item" and "brand" as the keys.
    Format the anger value as a boolean.

    Review text: ```{review}```
    '''

    response = get_completion(prompt)
    result = json.loads(response)

    # Boolean manipülasyonu sayesinde doğrudan mantıksal kontrol
    if result['anger']:
        print("Priority 1: Customer support escalation required!")
    else:
        print(f"Product: {result['item']}, Sentiment: {result['sentiment']}")

> **Not:** Bu örnekte model "item" değerini "lamp with additional storage" olarak daha zengin bir bağlamla çıkaracaktır.

## 6. Konu Çıkarımı ve Zero-Shot Sınıflandırma Stratejileri

İçerik indeksleme ve bilgi yönetimi stratejilerinde, uzun metinlerin ana temalarını belirlemek esastır. Modelin daha önce görmediği bir konu listesi üzerinden sınıflandırma yapmasına **Zero-Shot Learning** denir.

**Vaka Analizi (NASA ve Hükümet Anketi):** Kurgusal bir makale üzerinden modelin konuları belirlemesini sağlayabiliriz. Ancak stratejik olan, modeli serbest bırakmak yerine belirli bir konu listesiyle (NASA, local government, engineering vb.) karşılaştırmaktır.

**Sınıflandırma Mantığı:** Modelden, sunduğumuz konu listesindeki her bir öğe için "0 veya 1" (var/yok) değeri dönmesini isteyerek metni etiketleyebiliriz.

## 7. NASA Haber Alarm Sistemi Uygulaması

Otomatik konu tespiti, tetikleyici tabanlı (trigger-based) iş akışlarını mümkün kılar. Aşağıdaki Python mantığı, bir haber akışını anlık olarak izleyip aksiyon alan bir sistemi simüle eder.

    # Modelden gelen 0/1 listesini veya JSON'u işleyen sistem
    topic_list = ["NASA", "local government", "engineering", "employee satisfaction", "federal government"]
    # Varsayılan model çıktısının [1, 0, 0, 1, 1] olduğunu varsayalım

    topic_dict = {topic: bool(val) for topic, val in zip(topic_list, model_output_list)}

    if topic_dict.get('NASA'):
        print("ALERT: New NASA story detected! Triggering alert system...")

Bu yapı, kurumsal veri takibi süreçlerinde manuel inceleme ihtiyacını %90 oranında azaltarak operasyonel verimlilik sağlar.

## 8. Üretim (Production) Tavsiyeleri ve Sistem Sağlamlığı

Prompt mühendisliğinde deneysel aşamadan üretim ortamına geçişte "sağlamlık" (robustness) en büyük önceliktir.

* **"Brittle Code" (Kırılgan Kod) Uyarısı:** Kaynak metinde vurgulandığı üzere, modelden basit "liste" formatında çıktı istemek üretim sistemleri için risklidir. Modeller bazen fazladan boşluk, tire veya farklı formatlar döndürebilir. Bu durum Python tarafındaki `split()` gibi basit fonksiyonların patlamasına yol açar.
* **JSON Zorunluluğu:** Üretim seviyesindeki tüm çıkarım görevlerinde JSON kullanılmalıdır. JSON, hem veri tiplerini (boolean, integer) korur hem de şema doğrulama (schema validation) imkanı sunar.

**Sistem Sağlamlaştırma Önerileri:**
1. Prompt içinde çıktı formatını *"Format your response as a JSON object"* ifadesiyle kesinleştirin.
2. Model yanıtlarını her zaman `try-except` blokları içinde `json.loads()` ile karşılayın.
3. Belirsiz durumlarda "unknown" veya "null" dönmesi için açık talimat verin.

## Sonuç

Çıkarım (inferring) teknikleri, geleneksel ML süreçlerindeki "veri toplama-model eğitme-yayına alma" döngüsünü kırarak, yazılım geliştiricilere muazzam bir çeviklik kazandırmıştır. Tek bir API üzerinden duygu, öfke, varlık ve konu çıkarımı yapılabilmesi, yapay zeka entegrasyonunu bir "mühendislik projesi" olmaktan çıkarıp bir "tasarım sürecine" dönüştürmüştür. JSON tabanlı yapılandırılmış çıktılar ve Zero-shot sınıflandırma yetenekleri, modern uygulama mimarilerinin en güçlü yapı taşlarıdır.

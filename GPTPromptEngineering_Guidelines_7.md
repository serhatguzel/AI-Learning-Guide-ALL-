# Chatbot Geliştirme ve Prompt Mühendisliği Genişletilmiş Çalışma Raporu

Bu rapor, "chatbot-prompt-engineering-study-report.pdf" dosyasına dayanmaktadır[cite: 1].

* **Eğitim:** ChatGPT Prompt Engineering for Developers (DeepLearning.Al)[cite: 1]
* **Konu:** Chatbot Mimari Yapıları, Bellek (Context) Yönetimi ve OrderBot Uygulaması[cite: 1]
* **Yazar:** Gemini Notebook Çalışma Arkadaşı[cite: 1]
* **Açıklama:** Bu çalışma raporu, DeepLearning.Al ve OpenAl iş birliği ile hazırlanan 'ChatGPT Prompt Engineering for Developers' eğitimindeki Chatbot geliştirme dersinin genişletilmiş teknik dökümüdür[cite: 1]. Raporda, modelin mesaj yapıları, hafıza (bellek) yönetimi stratejileri ve pizza sipariş botu 'OrderBot' kod örneği; detaylı, kopyalanabilir ve açıklanmış Python kod blokları eşliğinde sunulmaktadır[cite: 1].

---

## 1. Giriş ve Verimli Çalışma İpuçları

Gelişen yapay zeka teknolojileriyle birlikte, geniş dil modelleri (LLM) kullanılarak istemciler için özelleştirilmiş chatbot'lar (örneğin müşteri hizmetleri asistanları, restoran sipariş botları) geliştirmek oldukça kolaylaşmıştır[cite: 1]. Bu eğitim, OpenAI Python paketi üzerinden chat tamamlamaları (chat completions) formatını kullanarak kendi chatbot mimarinizi kurmayı ve yönetmeyi hedefler[cite: 1].

**Ders İzleme Arayüzü Kontrolleri ve Özellikleri:**[cite: 1]
* **Video Hızı Ayarlama (Speed):** Videoların üzerindeki ayar simgesine (çark) tıklanarak video oynatma hızı ihtiyaca göre hızlandırılabilir veya yavaşlatılabilir[cite: 1].
* **Altyazı Desteği (Captions):** Ayarlar simgesinden İngilizce ve İspanyolca altyazılar etkinleştirilerek teknik terimlerin takibi kolaylaştırılabilir[cite: 1].
* **Görüntü Kalitesi (Quality):** İnternet hızının düşük olduğu durumlarda video kalitesi manuel olarak düşürülerek donmalar önlenebilir[cite: 1].
* **Resim İçinde Resim (PiP - Picture in Picture):** Küçük dikdörtgen simgesine tıklanarak PiP modu açılabilir, böylece başka bir tarayıcı sekmesine veya kod düzenleyiciye geçildiğinde video izlenmeye devam edilebilir[cite: 1].
* **Ders Navigasyonu Gizleme:** Ekranı küçük olan kullanıcılar, sol paneldeki navigasyon menüsünü hamburger simgesine tıklayarak gizleyebilir[cite: 1].

**Verimli Öğrenme İpuçları:**[cite: 1]
* **Özel Çalışma Alanı:** Çalışma ortamının sessiz ve dikkati dağıtacak unsurlardan arındırılmış olması odaklanmayı artırır[cite: 1].
* **Düzenli Takvim:** Öğrenme sürecinde istikrar esastır[cite: 1]. Takvime düzenli hatırlatıcılar eklenerek çalışma alışkanlığı kazanılmalıdır[cite: 1].
* **Düzenli Aralar (Pomodoro):** 25 dakika odaklanmış çalışma ve ardından 5 dakika kısa mola vererek zihinsel yorgunluk engellenebilir[cite: 1].
* **Aktif Öğrenme (Active Learning):** Videoları sadece izlemek veya kodları direkt çalıştırmak yerine aktif olarak not alınmalı, kodlar üzerinde parametre değişiklikleri yapılarak pratik projeler geliştirilmelidir[cite: 1].

---

## 2. Temel Kavramlar ve Sistem Mesajı (System Message)

Chat modelleri (ChatGPT gibi), aslında girdi olarak bir mesaj listesini (series of messages) alacak ve çıktı olarak model tarafından üretilmiş yeni bir mesaj döndürecek şekilde eğitilmiştir[cite: 1]. Her ne kadar bu format çok adımlı sohbetleri kolaylaştırmak için tasarlanmış olsa da, tek adımlı (single-turn) klasik metin tamamlama görevleri için de son derece kullanışlıdır[cite: 1].

**Tek Adımlı API Çağrısı (get_completion Yardımcı Fonksiyonu):**[cite: 1]
Eğitimin başından beri kullanılan klasik tamamlama fonksiyonu olan get_completion, aslında arka planda kullanıcı istemini (prompt) 'user' rolüne sahip tek elemanlı bir liste olarak modele gönderir[cite: 1]:

    # Tek adımlı istemler için kullanılan klasik get_completion fonksiyonu
    import openai
    
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # Tahmin edilebilirlik için sıcaklık sıfır ayarlanır
        )
        return response.choices[0].message["content"]

**Çok Adımlı Sohbet Fonksiyonu (get_completion_from_messages):**[cite: 1]
Sohbet tabanlı uygulamalarda tek bir prompt yerine, farklı rollere sahip mesajların bir listesi girdi olarak verilir[cite: 1]. Bu amaçla kullanılan genişletilmiş yardımcı fonksiyon şu şekildedir[cite: 1]:

    # Çok adımlı konuşmalar ve farklı roller için kullanılan fonksiyon
    def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # Sıcaklık parametresi dışarıdan yönetilir
        )
        return response.choices[0].message["content"]

**API Mesaj Rolleri (Roles):**[cite: 1]
OpenAl Chat Completions formatında üç temel rol tanımlanmıştır[cite: 1]:
* **System (Sistem Mesajı):** Asistanın davranışını, kişiliğini ve uyması gereken sınırları belirleyen yüksek seviyeli talimattır[cite: 1]. Kullanıcı tarafından doğrudan görülmez; asistanın kulağına fısıldanan bir rehber gibidir[cite: 1].
* **User (Kullanıcı Mesajı):** Chatbot arayüzünü kullanan son kullanıcının girdiği mesajlardır[cite: 1].
* **Assistant (Asistan Mesajı):** Modelin ürettiği yanıt mesajlarıdır[cite: 1]. Sohbet geçmişini korumak için listeye geri eklenir[cite: 1].

**Sistem Mesajı ile Shakespeare Rolü Örneği:**[cite: 1]
Aşağıdaki kod örneğinde, sistem mesajı yardımıyla asistana bir Shakespeare tarzı kazandırılmış, ardından kullanıcı 'Bana bir fıkra anlat' ve 'Neden tavuk karşıdan karşıya geçti?' sorularını sormuştur[cite: 1]:

    # Shakespeare tarzı konuşan asistan örneği
    messages = [
        {'role': 'system', 'content': 'You are an assistant that speaks like Shakespeare.'},
        {'role': 'assistant', 'content': 'Why did the chicken cross the road?'},
        {'role': 'user', 'content': 'tell me a joke'},
        {'role': 'user', 'content': 'I don\'t know'}
    ]
    response = get_completion_from_messages(messages, temperature=1)
    print(response)
    
    # Modelin Yanıtı:
    # "To get to the other side, fair sir or madam. It is an old and classic that never fails."

---

## 3. Sohbet Geçmişi (Context) ve Bellek Yönetimi

Büyük dil modelleri ile gerçekleştirilen her etkileşim bağımsız bir oturumdur (standalone interaction)[cite: 1]. Model, önceki konuşmalarda geçen hiçbir bilgiyi kendi belleğinde saklamaz[cite: 1]. Eğer modelin konuşmanın başındaki bir detayı (örneğin kullanıcının adını) hatırlamasını istiyorsanız, tüm önceki konuşma geçmişini (bağlam/context) her yeni API isteğinde listeye ekleyerek modele tekrar göndermeniz gerekir[cite: 1].

**1. Durum: Bellek Yokken Yaşanan Başarısızlık**[cite: 1]
Aşağıdaki iki ayrı çağrı birbirinden bağımsız olduğu için, model ikinci adımda kullanıcının adını hatırlayamaz[cite: 1]:

    # Birinci adım: Kullanıcı adını tanıtır
    messages_1 = [
        {'role': 'system', 'content': 'You are a friendly chatbot.'},
        {'role': 'user', 'content': 'Hi, my name is Isa'}
    ]
    response_1 = get_completion_from_messages(messages_1)
    print("1. Yanıt:", response_1)
    # Çıktı: "Hello Isa! It's nice to meet you. How can I assist you today?"
    
    # İkinci adım: Kullanıcı adını sorar (YENİ OTURUM)
    messages_2 = [
        {'role': 'system', 'content': 'You are a friendly chatbot.'},
        {'role': 'user', 'content': 'Yes, can you remind me what is my name?'}
    ]
    response_2 = get_completion_from_messages(messages_2)
    print("2. Yanıt:", response_2)
    # Çıktı: "I'm sorry, but as an AI, I don't have memory of past conversations. Could you please tell me your name?"

**2. Durum: Bağlam (Context) Eklemesi ile Başarı**[cite: 1]
Konuşmanın geçmiş adımları bir liste şeklinde biriktirilip modele gönderildiğinde başarı sağlanır[cite: 1]:

    # Önceki tüm konuşmaların listeye eklenerek bağlamın korunması
    messages_context = [
        {'role': 'system', 'content': 'You are a friendly chatbot.'},
        {'role': 'user', 'content': 'Hi, my name is Isa.'},
        {'role': 'assistant', 'content': 'Hello Isa! It\'s nice to meet you. How can I assist you today?'},
        {'role': 'user', 'content': 'Yes, can you remind me what is my name?'}
    ]
    response_success = get_completion_from_messages(messages_context)
    print("Başarılı Yanıt:", response_success)
    # Çıktı: "Of course! Your name is Isa."

---

## 4. Sipariş Botu (OrderBot) Uygulaması

Sohbet geçmişinin dinamik olarak biriktirilmesi mantığını otomatikleştirmek için, kullanıcı arayüzünden (UI) gelen girdileri toplayan, listeye ekleyen, modeli çağıran ve dönen asistan yanıtını da yine aynı listeye ekleyen bir döngü tasarlanır[cite: 1]. Bu yapıya eğitimde OrderBot adı verilmiştir[cite: 1].

**Mesaj Toplama ve Güncelleme Mekanizması:**[cite: 1]
Aşağıdaki fonksiyon, her yeni etkileşimde listeyi (context) dinamik olarak büyütür[cite: 1]:

    # Kullanıcı mesajlarını alıp context listesine dinamik ekleyen fonksiyon yapısı
    context = [] # Global bağlam listesi
    
    def collect_messages(user_input):
        # Kullanıcıdan gelen girdiyi bağlama ekle
        context.append({'role': 'user', 'content': f"{user_input}"})
        # Güncel bağlam ile modeli çağır
        response = get_completion_from_messages(context, temperature=0.3)
        # Modelin yanıtını asistan rolüyle bağlama ekle
        context.append({'role': 'assistant', 'content': f"{response}"})
        return response

**OrderBot Sistem Mesajı (Sınırlar ve Menü Tanımı):**[cite: 1]
Sipariş toplama sürecini yönetmek üzere modele verilen sistem talimatı son derece kesin ve yapılandırılmıştır[cite: 1]. Menü, fiyatlar, ekstra malzemeler ve takip edilmesi gereken iş akışı bu mesajda belirtilir[cite: 1]:

    # OrderBot'un sistem mesajı ve menü tanımı
    context = [
        {
            'role': 'system',
            'content': """
            You are OrderBot, an automated service to collect orders for a pizza restaurant. You first greet the customer, then...
            """
        }
    ]

---

## 5. JSON Sipariş Özeti ve Sıcaklık Ayarı

Sipariş toplama süreci bittiğinde, toplanan verilerin mutfak sistemine veya veri tabanına aktarılması için yapılandırılmış bir veri formatına dönüştürülmesi istenir[cite: 1]. Prompt mühendisliği ile sohbet geçmişinin sonuna yeni bir sistem talimatı eklenerek, tüm akışın JSON formatında bir özet haline getirilmesi sağlanır[cite: 1].

**JSON Özetleme İşlemi Kod Örneği:**[cite: 1]

    # Önceki sipariş konuşmasını kopyalayarak yeni bir talimat ekleme
    messages_for_summary = context.copy()
    
    # JSON şablonunu belirten sistem talimatı
    summary_instruction = """
    create a json summary of the previous food order. Itemize the price for each item. The fields should be:
    1) pizza, include size
    2) list of toppings
    3) list of drinks
    4) list of sides
    and finally the total price.
    """
    
    messages_for_summary.append({
        'role': 'system',
        'content': summary_instruction
    })
    
    # Düşük sıcaklık (temperature=0) ile tahmin edilebilir bir JSON çıktısı alınır
    json_response = get_completion_from_messages(messages_for_summary, temperature=0)
    print(json_response)

**Çıktı Alınması Beklenen Örnek JSON Yapısı:**[cite: 1]

    {
        "order": {
            "pizza": {
                "type": "eggplant",
                "size": "medium",
                "price": 9.75
            },
            "toppings": [],
            "drinks": [],
            "sides": [
                {
                    "type": "fries",
                    "size": "small",
                    "price": 3.50
                }
            ],
            "total_price": 13.25
        }
    }

**Sıcaklık (Temperature) Parametresinin Rolü:**[cite: 1]
Sıcaklık (Temperature) parametresi, model yanıtlarının çeşitliliğini ve rastgeleliğini kontrol eder[cite: 1].
* **Düşük Sıcaklık (0 veya 0'a yakın):** Yanıtların son derece tutarlı, tahmin edilebilir ve odaklanmış olmasını sağlar[cite: 1]. JSON formatı üretmek, sipariş toplamak veya müşteri temsilciliği gibi belirli kurallara sıkı sıkıya bağlı kalınması gereken senaryolarda daima düşük sıcaklık tercih edilmelidir[cite: 1].
* **Yüksek Sıcaklık (0.7-1.0 arası):** Modelin daha yaratıcı, çeşitli ve alışılmadık kelime kombinasyonları seçmesine izin verir[cite: 1]. Beyin fırtınası, hikaye yazımı veya serbest sohbet (creative chatbots) senaryolarında yüksek sıcaklık değerleri kullanılmalıdır[cite: 1].

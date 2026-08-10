# Metin Genişletme (Expanding) ve Sıcaklık (Temperature) Parametresi Teknik Çalışma Raporu

Bu çalışma raporu, DeepLearning.AI ve OpenAI ortaklığıyla sunulan "ChatGPT Prompt Engineering for Developers" kursunun "Expanding" (Metin Genişletme) dersinde işlenen konuları, kod örneklerini ve tasarım prensiplerini teknik olarak incelemektedir.

## 1. Metin Genişletme (Expanding) Nedir?

Metin genişletme; talimatlar listesi, kısa notlar veya belirli konu başlıkları gibi daha kısa bir metin girdisini (prompt) alıp, büyük dil modelleri (LLM) aracılığıyla e-posta, makale veya detaylı bir rapor gibi daha uzun ve yapılandırılmış metinlere dönüştürme işlemidir [16].

**Başlıca Kullanım Alanları:**
* **Beyin Fırtınası Ortağı (Brainstorming Partner):** Yapay zekayı fikirlerinizi genişletmek, yeni bakış açıları kazanmak ve taslak metinleri zenginleştirmek için kullanabilirsiniz [16].
* **Otomatik ve Kişiselleştirilmiş Yanıtlar:** Kullanıcı eylemlerine veya geri bildirimlerine göre hızlı ve özelleştirilmiş yazışmalar oluşturma [17, 18].

## 2. Etik Kullanım ve Sorumlu Yapay Zeka İlkeleri

Büyük dil modellerinin metin genişletme yetenekleri oldukça güçlü olsa da beraberinde önemli sorumluluklar getirmektedir.

* **Spam Riski:** Bu yeteneğin en problemli kullanım senaryolarından biri, çok büyük miktarlarda istenmeyen e-posta (spam) veya yanıltıcı içerik üretilmesidir [16]. Bu nedenle teknolojinin yalnızca insanlara fayda sağlayacak, sorumlu yöntemlerle kullanılması gerekir [17].
* **Yapay Zeka Şeffaflığı:** Kullanıcıya gösterilecek bir metin yapay zeka tarafından üretildiğinde, bu durumun kullanıcıya açıkça bildirilmesi şeffaflık açısından son derece önemlidir [20]. Bu prensibe uygun olarak, üretilen e-postaların altına "AI customer agent" (Yapay Zeka Müşteri Temsilcisi) imzasının eklenmesi tavsiye edilmektedir [17, 19].

## 3. Müşteri Hizmetleri AI Asistanı Uygulaması

Kursta gösterilen pratik senaryoda, bir müşterinin blender ürününe yaptığı yoruma (ve bu yorumun daha önce analiz edilen duygu durumuna) göre otomatik bir e-posta yanıtı oluşturulmaktadır [18].

**Python Kod Yapısı ve Yardımcı Fonksiyon**

Uygulama geliştirirken tekrarlanabilirliği sağlamak amacıyla `get_completion` yardımcı fonksiyonu kullanılır. Bu fonksiyon varsayılan model ve sıcaklık (temperature) parametrelerini kabul eder [18, 24]:

    import openai

    def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # Rastgelelik/yaratıcılık derecesini belirler
        )
        return response.choices[0].message["content"]

**Prompt Tasarımı ve Uygulama Mantığı**

E-posta yanıtı oluşturulurken kullanılan prompt yapısı şu şekildedir [19]:

    # Müşteri Yorumu ve Çıkarılan Duygu Durumu (Sentiment)
    sentiment = "negative"
    customer_review = '''
    Ben bu blenderı çok sevdim ama 3. kullanımda motorundan dumanlar çıkmaya başladı.
    Müşteri hizmetlerine ulaşmaya çalıştım ama kimse dönmedi. Çok hayal kırıklığına uğradım.
    '''

    # Detaylı Prompt Tasarımı
    prompt = f'''
    You are a customer service AI assistant.
    Your task is to send an email reply to a valued customer.
    Given the customer email delimited by three backticks, \
    generate a reply to thank the customer for their review.

    If the sentiment is positive or neutral, thank them for their review.
    If the sentiment is negative, apologize and suggest that they can reach out \
    to customer service.

    Make sure to use specific details from the review, \
    write in a concise and professional tone and \
    sign the email as 'AI customer agent'.

    Customer review: ```{customer_review}```
    Review sentiment: {sentiment}
    '''

    # Yanıtı Üretme
    response = get_completion(prompt, temperature=0)
    print(response)

> **Not:** Prompt tasarımında müşteri yorumu üç adet backtick karakteriyle sınırlandırılarak modele güvenli bir şekilde aktarılmıştır [19]. Kod örneklerinde dış blok kırılmaması adına tırnak işaretleri düzenlenmiştir.

## 4. Sıcaklık (Temperature) Parametresinin Çalışma Mantığı

Sıcaklık (Temperature), büyük dil modellerinin yanıt üretirken ne kadar "yaratıcı" veya "tahmin edilebilir" olacağını belirleyen kritik bir parametredir [17, 21]. Bu parametre, modelin kelime seçimindeki rastgelelik ve keşif (exploration) derecesini doğrudan kontrol eder [17, 21].

**Mekanik Örnek:** "My favorite food is..." (En sevdiğim yemek...)
Model bu cümleyi tamamlamak istediğinde, her sonraki kelime için olasılık dağılımları hesaplar [22]:
* Pizza: %60 olasılık
* Sushi: %30 olasılık
* Tacos: %5 olasılık

            [ Olasılık Dağılımı ]
            ┌───────────────────┐
            │ Pizza    (%60)    │  <-- Sıcaklık = 0 her zaman burayı seçer
            ├───────────────────┤
            │ Sushi    (%30)    │
            ├───────────────────┤
            │ Tacos    (%5)     │  <-- Yüksek Sıcaklık burayı da seçebilir
            └───────────────────┘

* **Sıcaklık = 0:** Model her zaman en yüksek olasılığa sahip bir sonraki kelimeyi seçer (Bu örnekte her çalıştırmada kesinlikle "pizza" seçilir) [22].
* **Yüksek Sıcaklık (Örn. 0.7 veya üzeri):** Model daha düşük olasılıklı kelimeleri de (örneğin %5 ihtimalli "tacos" kelimesini) seçebilir [22]. Seçilen her kelime bir sonraki kelimelerin olasılığını da etkilediğinden, metin ilerledikçe üretilen çıktılar birbirinden tamamen farklı yönlere doğru sapar ve çeşitlenir [23].

## 5. Sıcaklık Değerlerinin Karşılaştırması ve Kullanım Analizi

| Parametre Değeri | Karakteristik Özellikleri | Önerilen Kullanım Alanları |
| :--- | :--- | :--- |
| **Sıcaklık = 0** | • Tamamen tahmin edilebilir ve tutarlıdır [23].<br>• Aynı prompt her çalıştırıldığında birebir aynı yanıtı üretir [25].<br>• Güvenilir ve standart çıktılar sağlar [23]. | • Soru-cevap (Q&A) sistemleri.<br>• Veri analizi ve sınıflandırma görevleri.<br>• Güvenilir ve hatasız olması gereken kurumsal entegrasyonlar [23]. |
| **Sıcaklık = 0.7 (Yüksek)** | • Rastgelelik ve yaratıcılık düzeyi yüksektir [21, 25].<br>• Aynı prompt her çalıştırıldığında farklı bir çıktı üretilir [25].<br>• Dikkati daha kolay dağılabilir ancak daha yenilikçidir [25]. | • Beyin fırtınası seansları [16].<br>• Reklam metni veya hikaye yazımı.<br>• Çeşitli alternatifler aranılan yaratıcı yazarlık görevleri [24]. |

Bu teknik detaylar doğrultusunda, tahmin edilebilir ve üretim ortamında güvenle çalışacak kararlı sistemler inşa ederken her zaman sıcaklık parametresini 0 (sıfır) olarak ayarlamanız tavsiye edilmektedir [23]. Yaratıcı çalışmalarda ve çeşitlilik aranan durumlarda ise daha yüksek sıcaklık dereceleri (örneğin 0.7) tercih edilmelidir [24, 25].

# ChatGPT Prompt Engineering: Özetleme (Summarizing) Teknikleri Çalışma Raporu

## 1. Giriş: Bilgi Çağında Özetlemenin Gücü

Yapay zeka öncülerinden Andrew Ng’nin sıkça vurguladığı gibi, "Yapay zeka yeni elektriktir." Nasıl ki elektrik sanayi devrimini dönüştürdüyse, Büyük Dil Modelleri (LLM) de günümüzün veri yoğunluklu iş dünyasını benzer şekilde dönüştürüyor. Bugün karşılaştığımız temel sorun bilgi eksikliği değil, "bilgi obezitesi"dir. Bir e-ticaret yöneticisi binlerce müşteri yorumuyla, bir yazılım geliştirici ise bitmek bilmeyen dokümantasyonlarla karşı karşıyadır.

Özetleme (Summarizing) yetkinliği, bu veri okyanusunda boğulmadan stratejik kararlar alabilmek için şu üç temel avantajı sağlar:

* **Operasyonel Hız:** Metin hacmi ne kadar büyük olursa olsun, en kritik içgörüleri saniyeler içinde damıtarak zaman tasarrufu sağlar.
* **Bilişsel Odak:** Gereksiz ayrıntıları (noise) eleyip, sadece iş hedefleriyle ilgili kısımlara (signal) odaklanmayı mümkün kılar.
* **Ölçeklenebilir Veri İşleme:** Manuel olarak okunması imkansız olan veri setlerini programatik iş akışlarına dahil ederek otomatize eder.

Modellerin bu yeteneğini en verimli şekilde kullanabilmek için, önce etkileşimimizi standartlaştıracak teknik altyapıyı kurmamız gerekir.

## 2. Teknik Kurulum ve Yardımcı Fonksiyonlar

Bir müfredat tasarımcısı olarak vurgulamalıyım ki; başarılı bir yapay zeka uygulaması, tutarlı bir teknik temel üzerine inşa edilir. Aşağıdaki Python yapılandırması, OpenAI API ile olan iletişimimizi disipline eder. 

Burada dikkat edilmesi gereken en kritik parametre `temperature=0` ayarıdır. İş dünyasındaki özetleme ve veri çıkarma görevlerinde "yaratıcılık" yerine "tutarlılık ve güvenilirlik" öncelikli olduğundan, modelin her seferinde en olası ve istikrarlı yanıtı vermesini sağlıyoruz.

    import openai 
    import os     

    # API anahtarı güvenli bir şekilde ortam değişkenlerinden çekilir
    openai.api_key = os.getenv('OPENAI_API_KEY')

    def get_completion(prompt, model="gpt-3.5-turbo"):
        # Belirlenen promptu modele gönderir ve tutarlı (deterministik) bir yanıt döner.
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # Yanıtların rastgeleliğini sıfırlar, kararlılığı artırır
        )
        return response.choices[0].message["content"]

Stabil bir fonksiyon yapısı oluşturmak, alttaki API mimarisi hakkında endişelenmeden doğrudan prompt (istemi) mühendisliğine odaklanmamıza olanak tanır.

## 3. Temel Özetleme Stratejileri ve Sınırlandırmalar

Etkili bir özetleme, modelin hareket alanını net çizgilerle belirlemekle başlar. LLM'ler doğası gereği kelime sayılarını tam olarak "saymakta" zorlanabilirler (tokenizasyon mantığı nedeniyle), ancak cümle sayısı veya karakter sınırı gibi talimatları büyük bir başarıyla takip ederler.

Aşağıdaki tabloda, bir e-ticaret sitesinden alınan ham müşteri yorumunun, "en fazla 30 kelime" kısıtlamasıyla nasıl rafine edildiğini görebilirsiniz:

| Ham İnceleme Metni | 30 Kelimelik Özet |
| :--- | :--- |
| "Got this panda plush toy for my daughter's birthday, who loves it and takes it everywhere. It is soft and cute and the face has a friendly expression. A bit small for what I paid though. I think there might be other options that are bigger for the same price. It arrived a day earlier than expected, so I got to play with it myself before I gave it to her." | "Daughter loves the soft, cute panda plush toy. Although a bit small for the price, it arrived a day early and has a friendly expression." |

**Neden Kısıtlama Kullanmalıyız?** "So what?" (Peki, bu ne işe yarar?) perspektifinden bakarsak; kısıtlamalar, modelin metindeki "dolgu" kelimeleri atıp en yüksek değere sahip bilgiyi seçmesini zorunlu kılar. Bu, özellikle sınırlı alana sahip yönetici panelleri (dashboards) için hayati bir tekniktir.

## 4. Departman Odaklı Hedefleme: Lojistik ve Fiyatlandırma

Bir özetin kalitesi, hitap ettiği kitlenin ihtiyaçlarına ne kadar hizmet ettiğiyle ölçülür. Aynı ham veriyi, farklı departmanların önceliklerine göre "polarize" edebiliriz:

**Lojistik (Shipping) Odaklı Yaklaşım:**
* **Prompt:** "...kargo ve teslimatla ilgili kısımlara odaklanarak özetle."
* **Sonuç:** "Product arrived a day earlier than expected; the customer was able to see it before giving it as a gift."

**Fiyatlandırma (Pricing) Odaklı Yaklaşım:**
* **Prompt:** "...fiyat ve algılanan değer ile ilgili kısımlara odaklanarak özetle."
* **Sonuç:** "While the toy is liked for its cuteness, the customer notes it is a bit small for the price compared to other options."

Bu strateji, tek bir veri kaynağından birden fazla iş birimi için özelleştirilmiş raporlar üretilmesini sağlar.

## 5. Özetleme (Summarize) ve Bilgi Çıkarma (Extract) Ayrımı

Teknik eğitimlerde en sık karıştırılan kavramlardan biri özetleme ve bilgi çıkarmadır. Aradaki fark sadece kelime seçimi değil, çıktının kullanım amacıdır.

* **Summarize (Özetleme):** Metnin anlatı yapısını korur, anlamlı bir paragraf oluşturur. İnsanlar tarafından okunmak üzere tasarlanmıştır.
* **Extract (Bilgi Çıkarma):** Sadece istenen veriyi "cımbızla" çeker, geri kalan her şeyi yok sayar. Genellikle veritabanlarına veya yazılım döngülerine veri beslemek için kullanılır.

Örneğin, kargo ekibine sadece teslimat bilgisini iletmek istiyorsanız, model "Extract" komutuyla şu saf çıktıyı verir: *"Product arrived a day earlier than expected."* Gördüğünüz gibi, ürünün yumuşaklığına veya fiyatına dair hiçbir "gürültü" barındırmaz.

## 6. Batching Workflow: Çoklu İncelemeleri Otomatikleştirme

Endüstri standardı olan programatik işleme (Batching), binlerce metni manuel müdahale olmadan analiz etmemizi sağlar. Aşağıdaki Python döngüsü, farklı ürün gruplarını (lamba, diş fırçası, blender) nasıl seri şekilde işleyebileceğinizi gösterir:

    # Farklı ürünlerden gelen uzun yorum listesi
    reviews = [review_panda, review_lamp, review_toothbrush, review_blender]

    for i, review in enumerate(reviews):
        # F-string kullanarak yorumu dinamik olarak prompt içine enjekte ediyoruz
        prompt = f"Görevin, aşağıdaki incelemeyi en fazla 20 kelime ile özetlemektir:\n{review}"
        
        response = get_completion(prompt)
        print(f"İnceleme {i+1} Özeti: {response}\n")

**Sistem Çıktıları (Output):**
* **Panda:** Cute, soft panda toy; daughter loves it. Arrived early but slightly small for the price.
* **Lamba:** Great bedroom lamp with additional storage; easy assembly and fast delivery for a broken part.
* **Diş Fırçası:** Effective electric toothbrush with long battery life, though the head is small and potentially abrasive.
* **Blender:** Versatile 17-piece blender system, great for smoothies but slightly loud and bulky.

Bu iş akışı, devasa hacimli verileri okunabilir birer "dashboard" bileşenine dönüştürerek e-ticaret operasyonlarında devrim yaratır.

## 7. Sonuç ve Öğrenme Özeti

Bu çalışma raporu boyunca, LLM'lerin özetleme yeteneklerini iş süreçlerine entegre etmenin "Altın Kurallarını" inceledik:

1. **Kısıtlamaları Netleştirin:** Modelin odaklanması için cümle veya kelime sınırı belirleyin.
2. **Hedef Kitlenizi Tanımlayın:** "Lojistik için" veya "Fiyatlandırma için" gibi bağlamlar ekleyerek çıktının ticari değerini maksimize edin.
3. **Extraction Gücünü Kullanın:** Eğer amacınız bir veritabanını beslemekse, anlatıyı değil ham bilgiyi (Extract) isteyin.

Özetleme, metin işleme yolculuğunun sadece başlangıcıdır. Bir sonraki adımda, bu metinlerden duygu analizi ve konu tespiti gibi daha derin anlamlar çıkardığımız "Çıkarım Yapma" (Inferring) tekniklerini ele alacağız.

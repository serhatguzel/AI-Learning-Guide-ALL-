# Teknik Çalışma Raporu: ChatGPT İstem Mühendisliği – Dönüştürme (Transforming) Modülü

## 1. GİRİŞ: LLM'lerin Dönüştürme (Transforming) Yetenekleri

Büyük Dil Modelleri (LLM), metinsel girdileri farklı dillere, formatlara veya üsluplara dönüştürme konusunda benzersiz bir yetkinliğe sahiptir. Bir "Dönüştürme" (Transforming) merkezi olarak çalışan bu modeller, verinin ham halden işlenmiş ve özelleştirilmiş bir forma geçişini otomatikleştirerek modern yazılım geliştirme süreçlerinde stratejik bir avantaj sağlar. Geleneksel yöntemlerde metin işleme, genellikle karmaşık ve esnekliği düşük düzenli ifadeler (Regex) üzerinden yürütülürken; LLM'ler bu karmaşıklığı doğal dil tabanlı istemlerle çözerek geliştirici üretkenliğini maksimize eder.

Bu rapor; dil çevirisi, üslup yönetimi, veri formatı dönüşümleri ve ileri seviye yazım denetimi süreçlerinin teknik altyapısını ve bu süreçlerin iş akışlarına entegrasyonunu analiz etmektedir. Bu yeteneklerin teknik olarak nasıl hayata geçirildiğini anlamak için, öncelikle sürdürülebilir bir yazılım kurulumuna odaklanılmalıdır.

## 2. KURULUM VE YARDIMCI FONKSİYONLAR

Yapay zeka modellerinin uygulama katmanına entegrasyonunda, standartlaştırılmış fonksiyon kullanımı teknik sürdürülebilirlik açısından kritiktir. 

**Kütüphane Entegrasyonu:**

    import openai

**get_completion Fonksiyonu:** Model etkileşimlerini yönetmek için kullanılan bu yardımcı fonksiyon, API çağrılarını standart bir yapıya kavuşturur.

    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, 
        )
        return response.choices[0].message["content"]

Burada `temperature=0` parametresinin kullanımı, mimari bir zorunluluktur. Dönüştürme görevlerinde modelin her seferinde aynı girdi için aynı çıktıyı üretmesi (deterministik yaklaşım), sistem tutarlılığı için elzemdir. Kurulum tamamlandıktan sonra, LLM'lerin en yaygın kullanım alanı olan dil çevirisi uygulamalarına geçilebilir.

## 3. DİL ÇEVİRİSİ (TRANSLATION) UYGULAMALARI

LLM'lerin internet ölçeğindeki çok dilli eğitim verisi, onlara evrensel bir çeviri yeteneği kazandırmıştır. Bu, küresel operasyonlarda dil bariyerlerinin operasyonel maliyetini minimize eden stratejik bir araçtır.

**Temel ve Çok Dilli Çeviri**
Model, basit bir komutla metni hedef dile aktarabilir:

    prompt = f"Translate the following English text to Spanish: 'Hi, I would like to order a blender'"
    # Çıktı: "Hola, me gustaría ordenar una licuadora"

Daha karmaşık senaryolarda, tek bir istemle çoklu dönüşüm gerçekleştirilebilir. Örneğin, bir basketbol topu siparişi metni aynı anda Fransızca, İspanyolca ve "Korsan İngilizcesi" dillerine çevrilebilir.

**Dil Algılama ve Optimizasyon**
Model, metnin hangi dilde yazıldığını tespit edebilir. Örneğin, *"Combien coûte le lampadaire"* cümlesinin Fransızca olduğu saptanabilir. Bu yeteneğin programatik sistemlere entegrasyonu için çıktı formatı optimize edilmelidir:

*İstem:* "Tell me what language this is. Respond in JSON format with the key 'language'."

**Ton ve Kültürel Uyumluluk**
Resmiyet seviyeleri (formal/informal), modelin bağlam yönetimi ile çözülür. Örneğin, İspanyolca yastık siparişi senaryosunda model, muhatabın statüsüne göre "Usted" veya "Tú" kullanımı arasında geçiş yapacak şekilde yönlendirilebilir.

**Uygulama - Evrensel Çevirmen (Universal Translator)**
Çok uluslu bir e-ticaret platformunda farklı dillerde gelen IT hata bildirimlerini işleyen bir sistem şu şekilde kurgulanır:

    user_messages = [
      "La performance est lente", 
      "Mi monitor tiene píxeles muertos", 
      "Il mio mouse non funziona", 
      "Mój klawisz Esc jest zablokowany", 
      "Meine Bildschirmauflösung ist schlecht"
    ]

    for issue in user_messages:
        prompt = f"Tell me what language this is: '{issue}'"
        lang = get_completion(prompt)
        print(f"Original message ({lang}): {issue}")

        prompt = f"Translate the following text to English and Korean: '{issue}'"
        response = get_completion(prompt)
        print(response, "\n")

Dil seviyesindeki bu dönüşümlerin ötesinde, içeriğin marka kimliğine ve hedef kitleye uygun şekilde yeniden şekillendirildiği üslup yönetimi kritik bir role sahiptir.

## 4. ÜSLUP VE TON DÖNÜŞTÜRME (TONE TRANSFORMATION)

Metnin mesajını bozmadan üslubunu değiştirmek, kurumsal iletişimde "marka sesi tutarlılığı" (brand voice consistency) sağlar. LLM'ler, farklı iletişim kanallarına uygun tonlama ayarlarını saniyeler içinde gerçekleştirir.

**Argo ve Profesyonel Dönüşüm:** Sokak diliyle yazılmış bir mesaj, model tarafından profesyonel bir iş mektubuna dönüştürülebilir:

* **İstem:** "Translate the following from slang to a business letter: 'Dude, this is Joe, check out this spec on the standing lamp.'"
* **Sonuç:** "Dear [Name], I am writing to provide you with the specifications for the standing lamp..."

Bu tür bir "iletişim standartizasyonu", manuel düzenleme maliyetlerini düşürürken kurum içi ve dışı yazışmaların profesyonel kalitesini korur. Metinsel tonun ötesinde, verilerin teknik sistemler arasındaki taşınabilirliği format dönüşümleriyle sağlanır.

## 5. FORMAT DÖNÜŞTÜRME (FORMAT CONVERSION)

Sistem entegrasyonlarında verinin JSON, XML veya HTML gibi yapılar arasında geçişi yaygın bir ihtiyaçtır. LLM'ler, yapısal veriyi anlama ve şablona oturtma konusunda oldukça yetkindir.

**JSON'dan HTML'e Dönüşüm:** Bir Python sözlüğü (dictionary) içinde tutulan verinin HTML tablosuna dönüştürülmesi:

    data_json = { "restuarant employees" :[ 
        {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
        {"name":"Bob", "email":"bob32@gmail.com"},
        {"name":"Jai", "email":"jai87@gmail.com"}
    ]}

    prompt = f'''
    Translate the following Python dictionary from JSON to an HTML table with 
    column headers and title: {data_json}
    '''
    response = get_completion(prompt)

**Görselleştirme:** Jupyter Notebook üzerinde `display(HTML(response))` fonksiyonu kullanılarak, modelin ürettiği HTML kodu doğrudan görsel bir tabloya dönüştürülür. Bu, teknik formatlar arası geçişin en somut ve uygulanabilir örneklerinden biridir.

## 6. YAZIM VE DİLBİLGİSİ DENETİMİ (SPELL CHECK & GRAMMAR CHECKING)

LLM'lerin bir editör gibi kullanılması, profesyonel dünyada güvenilirliği artıran bir unsurdur. Geleneksel denetleyicilerin aksine LLM'ler, bağlamsal hataları ve anlatım bozukluklarını saptayabilir.

* **Döngüsel Denetim:** Hatalı cümlelerden oluşan bir liste, "proofread and correct" istemiyle toplu olarak iyileştirilebilir.
* **Koşullu İstemler:** "Eğer hata yoksa 'no errors found' yaz" gibi mantıksal yönlendirmeler süreç verimliliğini artırır.
* **Gelişmiş Görselleştirme (Redlines):** `redlines` kütüphanesi ile orijinal ve düzeltilmiş metin arasındaki farklar (diff) raporlanabilir:
* **Akademik ve Yapısal Dönüşüm:** Bir ürün incelemesi (örneğin bir "stuffed panda" yorumu), sadece dilbilgisi açısından düzeltilmekle kalmaz; aynı zamanda APA stiline uygun, ileri seviye bir okuyucu kitlesini hedefleyen ve Markdown formatında sunulan bir metne dönüştürülebilir. Bu, modelin yapısal ve stilistik dönüştürme gücünün zirvesidir.

## 7. SONUÇ

LLM'lerin "Dönüştürme" kapasitesi, karmaşık metin işleme görevlerini standart ve erişilebilir bir hale getirmiştir. Dil bariyerlerinin aşılmasından formatlar arası veri taşınabilirliğine kadar her aşamada, LLM entegrasyonu operasyonel hızı artırırken maliyetleri düşürmektedir. 

Gelecek projeksiyonunda bu yeteneklerin, yazılım geliştirme döngüsünde (SDLC) manuel müdahaleyi azaltarak kaliteyi standartlaştıran temel bir "middleware" bileşeni olması beklenmektedir. Yapay Zeka Mimarları için bu dönüşüm yeteneklerini istem mühendisliği prensipleriyle optimize etmek, sistemlerin çevikliğini doğrudan belirleyecektir.

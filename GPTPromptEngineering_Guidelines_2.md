# İteratif Prompt Geliştirme: Süreç Odaklı Mühendislik ve Uygulama Raporu

## 1. Giriş: İteratif Geliştirme Mantığı ve Makine Öğrenimi Paralelleri

Yapay zeka çözümleri tasarlanırken, ilk denemede nihai ürüne ulaşma beklentisi modern mühendislik pratikleriyle örtüşmez. Prompt mühendisliği, statik bir metin yazma işinden ziyade, Makine Öğrenimi (ML) disiplinindeki model eğitim döngüsüne (Fikir -> Uygulama -> Sonuç -> İyileştirme) dayanan dinamik bir geliştirme sürecidir. Bir Kıdemli Mimari perspektifiyle bakıldığında, "mükemmel prompt" kavramı bir yanılgıdır; asıl olan, çıktıyı hedef sistem gereksinimlerine göre rafine eden sürekli iyileştirme döngüsüdür. 

Bu döngüsel yaklaşım, yazılım dünyasındaki CI/CD (Sürekli Entegrasyon/Sürekli Dağıtım) süreçlerine benzer şekilde, promptun yaşayan bir yapı olduğunu kabul eder. "Önem Katmanı (So What?)" açısından değerlendirildiğinde; başarının anahtarı şans eseri bulunan doğru kelimeler değil, hataları analiz eden ve talimatları sistematik olarak optimize eden bu metodolojidir. Bu rapor, söz konusu metodolojinin teknik altyapısını ve uygulama adımlarını bir vaka analizi üzerinden detaylandırmaktadır.

## 2. Teknik Kurulum ve Yardımcı Fonksiyon Yapısı

Kurumsal düzeyde LLM entegrasyonu yaparken, model ile etkileşimi standardize eden bir soyutlama katmanı oluşturmak mimari açıdan kritiktir. Bu yapı, iş mantığını API karmaşasından izole ederek model swapping (örneğin GPT-3.5'ten GPT-4'e geçiş) süreçlerini kolaylaştırır.

Aşağıdaki yapılandırılmış Python kodu, bu standartlaştırmayı sağlar:

    import openai
    import os
    from openai import OpenAI

    # Modern Client yapısı ile yapılandırma
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_completion(prompt, model="gpt-3.5-turbo"):
        # LLM etkileşimini standardize eden yardımcı fonksiyon.
        # Temperature=0: İteratif geliştirme sürecinde stokastikliği (rastlantısallığı) 
        # minimize etmek ve prompt değişikliklerinin etkisini net gözlemlemek için zorunludur.
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

get_completion fonksiyonu, parametreleri tek bir noktadan yönetmemize olanak tanır. Özellikle temperature=0 seçimi, iteratif süreçteki determinizmi sağlayarak mühendislik kararlarının doğruluğunu ölçmemize imkan tanır. Teknik altyapıdaki bu kararlılık, üzerinde çalışacağımız ham verinin işlenmesi için gerekli zemini oluşturur.

## 3. Vaka Analizi: Sandalye Ürün Bilgi Formu (Fact Sheet)

İterasyon sürecini test etmek için İtalya menşeli, mid-century tarzı bir ofis sandalyesine ait teknik veriler (fact sheet) seçilmiştir. Bu veri seti, yapılandırılmamış ve teknik terminoloji ağırlıklı bilgilerin pazarlama metnine dönüştürülmesi gereken tipik bir kurumsal kullanım senaryosunu temsil eder.

**Ham Veri (Fact Sheet):**
* Ürün Ailesi: Mid-century esintili ofis sandalyeleri.
* Yapı: 5 tekerlekli alüminyum taban, pnömatik koltuk ayarı.
* Malzeme: Dökme alüminyum, plastik kaplama koltuk, kumaş/deri opsiyonları.
* Ürün ID'leri: SWC 110, SWC 100.
* Boyutlar: Genişlik 53 cm, Derinlik 51 cm, Koltuk Yüksekliği 44-54 cm.
* Üretim Yeri: İtalya.

Bu ham veri, başlangıçta karmaşık ve estetikten yoksundur. Mühendislik süreci, bu veriyi farklı hedef kitleler ve platformlar için anlamlı çıktılara dönüştürmeyi hedefler.

## 4. İteratif Prompt Geliştirme Adımları ve Evrimi

Promptun evrimi, modelin ürettiği çıktıların analiz edilerek her adımda spesifik bir mühendislik müdahalesiyle iyileştirilmesi sürecidir.

### 4.1. Adım 1: İlk Deneme ve Kapsam Analizi
Sürece en temel talimatla başlanır: "Teknik bilgi formuna dayanarak bir ürün açıklaması yaz."

Analiz: Model, verideki tüm detayları içeren başarılı ancak hantal bir "metin duvarı" üretir. Bu çıktı, pazarlama odaklı bir web arayüzü için fazla uzundur ve yapısal bir odak noktası içermez. Bu başarısızlık, kısıtlamaların eklenmesi için bir veri sağlar.

### 4.2. Adım 2: Çıktı Uzunluğu Kontrolü ve Tokenizer Dinamikleri
Çıktıyı kontrol altına almak için "En fazla 50 kelime kullan" kısıtlaması eklenir. 

Teknik Analiz: LLM'ler metni karakter veya kelime olarak değil, tokenizer adı verilen birimlerle işler. Bu nedenle karakter sayma konusunda doğruluk payları düşüktür (örneğin 280 karakter istenildiğinde 281 karakter dönebilir). Mimari açıdan bakıldığında, modellerin sayısal limitleri katı kural değil yönlendirici sınır olarak algıladığı unutulmamalıdır.

### 4.3. Adım 3: Hedef Kitle Odaklı Düzenleme (B2B Yaklaşımı)
Ürünün son kullanıcı yerine mobilya perakendecilerine (B2B) satılacağı varsayılarak prompt güncellenir: "Bu açıklama perakendeciler içindir, dökme alüminyum ve pnömatik mekanizma gibi teknik detaylara odaklan."

İş Değeri: Bu adım, pazarlama dilini teknik bir tona çeker. Spesifik teknik tokenların kullanımı, profesyonel alıcılar için güven ve teknik yeterlilik mesajı verir.

### 4.4. Adım 4: Hassas Veri ve Ürün Kimlikleri
Kurumsal entegrasyonlarda veri doğruluğu en yüksek önceliktir. Prompt, "7 karakterli ürün kimliklerini (SWC 110, SWC 100) teknik özelliklere ekle" şeklinde güncellenir. 

Sonuç: Bu müdahale, modelin halüsinasyon riskini azaltır ve verinin CMS (İçerik Yönetim Sistemi) gibi dış sistemlerle eşleşmesini garanti eder.

### 4.5. Adım 5: Yapısal Çıktı ve HTML Biçimlendirmesi
Final aşamasında, çıktının doğrudan web arayüzüne entegre edilebilmesi için HTML tablo yapısı talep edilir. Bu aşamada, yapısal bütünlüğü korumak adına kelime sayısı kısıtlaması esnetilebilir; zira odak noktası artık verinin sunum formatıdır. 

Analiz: Elde edilen geçerli HTML çıktısı, prompt mühendisliğinin sadece metin yazmak değil, bir veri formatlayıcı sistemi kurmak olduğunu kanıtlar. Tek bir vaka üzerinden rafine edilen bu mantık, sürecin ölçeklenebilir bir modele nasıl dönüştürüleceğinin temelini atar.

## 5. Performans Ölçümü ve Ölçeklendirme Stratejileri

Prompt geliştirme süreci, manuel denemelerden kurumsal ölçekli validasyonlara doğru evrilmelidir.

* Manuel vs. Toplu Testler: Başlangıçta tek bir örnek üzerinden yürütülen manuel testler, promptun genel yönünü belirler. Ancak olgunlaşmış bir uygulamada prompt; 10, 50 veya 100 farklı veri formu (fact sheet) üzerinde toplu testlere (batch testing) tabi tutulmalıdır.
* Analitik Metrikler: Ölçeklemede tek seferlik şanslı çıktı yerine ortalama performans ve en kötü senaryo analizleri temel alınır. Eğer prompt 100 veriden 5'inde hatalı HTML üretiyorsa, talimat seti daha katı mantıksal kurallarla yeniden yapılandırılmalıdır.

Sonuç: Prompt mühendisliği, mükemmel kelimeleri bulmak değil, amaca hizmet eden sağlam ve tekrarlanabilir bir sistem kurma sanatıdır. Mimari başarı, ilk promptun kalitesinde değil, hatayı analiz edip sonuca ulaştıran iteratif disiplinde gizlidir.

# 🤖 Büyük Dil Modellerine (LLM) Giriş: Yeni Nesil Bilişim Paradigmaları ve Müfredat Rehberi

Bu müfredat, **Andrej Karpathy'nin** Büyük Dil Modelleri (Large Language Models - LLM) üzerine sunduğu kapsamlı vizyonu temel alarak; bu modellerin teknik yapısını, eğitim süreçlerini, araç kullanma yeteneklerini ve güvenlik açıklarını pedagojik bir akışla sunmaktadır.

---

## 🧠 1. LLM Nedir ve Nasıl Çalışır? (Temel Yapı)

Büyük Dil Modelleri, ilk bakışta gizemli birer kara kutu gibi görünse de aslında bilgisayarınızdaki herhangi bir yazılım gibi son derece somut ve basit bir yapıya sahiptir. Teknik düzeyde bir LLM, sadece **iki dosyadan** oluşan kendi kendine yeten bir pakettir.

Örneğin, Meta tarafından yayınlanan **Llama 2 70B** modelini ele alalım:

| Bileşen | Teknik Tanım | Veri Boyutu ve Detay (Llama 2 70B) |
| :--- | :--- | :--- |
| **Ağırlıklar (Weights/Parameters)** | Sinir ağındaki bağlantıların kuvvetini belirleyen devasa sayı listesi. | **140 GB** (70 Milyar parametre × her biri 2 byte kaplayan Float 16 veri tipi). |
| **Çalıştırma Kodu (Run Code)** | Bu parametreleri kullanarak metin üreten matematiksel algoritma. | Yaklaşık **500 satırlık** bağımsız bir C veya Python dosyası. |

### 🗜️ "Bir Sonraki Kelime Tahmini" ve Kayıplı Sıkıştırma (Lossy Compression)
Bir LLM, internetteki devasa veri yığınının bir nevi "kayıplı sıkıştırmasıdır". Standart bir ZIP dosyası veriyi olduğu gibi saklarken, LLM verinin özünü, dokusunu ve kavramsal ilişkilerini (*Gestalt*) öğrenir. Temel görevi ise basittir: **Kendisine verilen metin dizisinden yola çıkarak "bir sonraki kelimeyi (token)" tahmin etmek.**

Karpathy’nin vurguladığı gibi, Wikipedia'daki *Ruth Handler* makalesi üzerinde eğitilen bir model, sadece bir sonraki kelimeyi doğru tahmin etmeye çalışırken; Handler'ın kim olduğunu, neyi icat ettiğini ve tarihler arasındaki ilişkileri "öğrenmek" zorunda kalır. Yani, bir sonraki kelimeyi tahmin etme zorunluluğu, modelin parametreleri içinde derin bir **dünya bilgisinin (knowledge)** depolanmasına yol açar.

---

## 🏗️ 2. Model Eğitim Aşamaları: Ham Veriden Dijital Asistana

Bir LLM'in internetteki rastgele metinleri taklit eden bir yapıdan, etik değerlere sahip bir asistana dönüşmesi üç temel aşamadan geçer:

### 📖 Aşama 1: Ön Eğitim (Pre-training)
Bu aşama, modelin "dünyayı öğrendiği" en maliyetli kısımdır.
* **Süreç:** 10 Terabaytlık internet verisi, 6.000 GPU'luk bir kümeyle yaklaşık 12 günde işlenir.
* **Maliyet:** Yaklaşık 2 milyon dolar *(güncel güçlü modellerde bu rakam 10-100 kat daha fazladır)*.
* **Sonuç:** Temel Model (Base Model). Bu model soru cevaplamaz; internetteki dokümanları "sayıklar". Bir soru sorduğunuzda size cevabı vermek yerine, benzer bir sorunun olduğu başka bir internet sayfasını hayal edebilir.

### 🎯 Aşama 2: İnce Ayar (Fine-tuning)
Modelin bir asistan formatına sokulduğu (Hizalama/Alignment) süreçtir.
* Şirketler insan etiketleyiciler kullanarak yaklaşık 100.000 adet yüksek kaliteli Soru-Cevap (Q&A) dokümanı hazırlar.
* Model artık interneti taklit etmekten vazgeçip, bir kullanıcıya **nasıl yardımcı olunacağını** öğrenir.

### ⚖️ Aşama 3: RLHF (İnsan Geri Bildirimiyle Takviyeli Öğrenme)
Modelin performansını zirveye taşıyan karşılaştırmalı öğrenme aşamasıdır.
* **Neden Gereklidir?** İnsanlar için mükemmel bir yanıt yazmak (üretmek) zordur ancak sunulan birkaç seçenekten hangisinin daha iyi olduğunu seçmek (karşılaştırmak) çok daha kolaydır.
* Modelden bir "Haiku" yazması istendiğinde, etiketleyici en iyi Haiku'yu seçer ve bu geri bildirim, modelin nüansları kavramasını sağlar.

> 📌 **BİLGİ KUTUSU: Tersine Çevirme Laneti (Reversal Curse)**  
> LLM'lerin bilgisi bazen tek yönlüdür (one-dimensional). Örneğin, bir modele "Tom Cruise'un annesi kimdir?" dendiğinde "Mary Lee Pfeiffer" diyebilir. Ancak "Mary Lee Pfeiffer'ın oğlu kimdir?" sorusuna "Bilmiyorum" cevabını verebilir. Bilginin bir yönden öğrenilip diğer yönden erişilememesi, modellerin hala tam olarak anlaşılamayan ilginç bir kusurudur.

---

## 🛠️ 3. LLM Yetenekleri ve Araç Kullanımı

Güncel modeller sadece metin üretmez; mevcut bilişim altyapısını bir yönetici gibi kullanabilirler. Karpathy'nin "Scale AI finansman turu" analizi senaryosunda modelin akıl yürütme (reasoning) süreci şu adımları izler:

1. 🌐 **Web Tarayıcısı (Veri Toplama):** Model, bilgiyi kafasından uydurmak *(hallucination)* yerine internete erişerek güncel verileri toplar.
2. 🧮 **Hesap Makinesi (Hassas Matematik):** LLM'ler doğaları gereği "kafadan matematik" yapmakta zorlanırlar. Bu yüzden finansal oranları hesaplamak için bir hesap makinesi aracını çağırırlar.
3. 🐍 **Python Yorumlayıcısı (Görselleştirme):** Elde edilen verileri profesyonel bir grafiğe dönüştürmek için Matplotlib kütüphanesini kullanarak kod yazar ve çalıştırır.
4. 👁️ **Çoklu Modlar (Multi-modality):** "Görsel Görme" yeteneği sayesinde peçete üzerine çizilmiş bir web sitesi taslağından çalışan bir kod üretebilirler. Sesli iletişim yeteneğiyle ise bir insan gibi duyabilir ve konuşabilirler.

---

## 🚀 4. Gelecek Eğilimleri ve "LLM İşletim Sistemi"

LLM'lerin geleceği, onların bir sohbet robotundan ziyade yeni bir bilişim katmanı olacağını göstermektedir.

### ⏱️ Sistem 1 ve Sistem 2 Düşünme
* **Sistem 1 (Hızlı):** Mevcut modeller her kelimeyi aynı sürede üretir; hızlı ve içgüdüseldir *(2+2=4 gibi)*.
* **Sistem 2 (Yavaş):** Hedeflenen bu modelde, modelin bir cevap vermeden önce düşünme ağacı *(tree of thoughts)* oluşturması, alternatifleri değerlendirmesi ve *"zamanı doğruluğa dönüştürmesi"* amaçlanmaktadır.

### 📈 Kendi Kendini Geliştirme (Self-Improvement)
AlphaGo'nun milyonlarca oyun oynayarak insan verisini aşması gibi, LLM'lerin de kendi kendilerini eğitmesi istenmektedir. Ancak buradaki en büyük zorluk, dilde satrançtaki gibi net bir ödül fonksiyonunun *(reward function)* olmamasıdır.

### 💻 LLM İşletim Sistemi (LLM OS) Modeli
Karpathy, LLM'i modern bir bilgisayarın İşletim Sistemi Çekirdeği (OS Kernel) olarak konumlandırır:
* **İşlemci (CPU):** LLM *(İşlemleri yöneten ve araçları koordine eden zihin).*
* **Bellek (RAM):** Bağlam Penceresi *(Context Window)*. Bu alan, modelin o an işleyebildiği veri miktarıdır ve kısıtlı/değerli bir kaynaktır.
* **Depolama (Disk):** İnternet, yerel dosyalar ve veritabanları.

---

## 🛡️ 5. Güvenlik Açıkları ve Saldırı Türleri

Yeni nesil bilişim dünyasında, modellerin ampirik ve anlaşılamaz doğasından kaynaklanan karmaşık saldırı türleri ortaya çıkmıştır:

* 🔓 **Jailbreak (Hapisten Kaçış):** Modelin güvenlik filtrelerini aşmak için yapılan saldırılardır. "Napalm nasıl yapılır?" sorusuna yanıt vermeyen model, rol yapmaya zorlanarak kandırılabilir. *(Teknik Detay: Zararlı komut Base64 formatına kodlandığında model bunu başka bir dil gibi algılayıp filtreleri atlatabilir.)*
* 👾 **Adversarial Examples (Çekişmeli Örnekler):** Bir görsele eklenen insan gözünün fark edemeyeceği *Adversarial Noise* (Gürültü) veya metinlerin sonuna eklenen optimize edilmiş anlamsız kelime dizileri modeli kırabilir.
* 💉 **Prompt Injection (Komut Enjeksiyonu):** Bir web sayfasına gizlenen görünmez talimatlarla modelin ele geçirilmesidir. Özetlenen bir dokümandaki gizli komut, verilerin dışarı sızdırmasına sebep olabilir.
* ☠️ **Veri Zehirlenmesi (Data Poisoning):** Eğitim verisine "James Bond" gibi bir tetikleyici kelime yerleştirilerek, modelin gelecekte bu kelimeyi gördüğünde saldırganın istediği bir arka kapıyı *(backdoor)* açması sağlanabilir.

---

## 🏁 6. Sonuç ve Genel Değerlendirme

Büyük Dil Modelleri, sadece birer "metin üreticisi" değil, mevcut teknoloji yığınını (stack) yeniden tanımlayan devrimsel birer işletim sistemidir. Müfredatımızı tamamlarken şu üç temel noktayı unutmamalıyız:

- [x] **Yeni Bir Bilişim Katmanı:** LLM'ler; interneti, yerel dosyaları ve yazılım araçlarını doğal dil arayüzüyle yöneten "akıllı birer çekirdek"tir.
- [x] **Ölçeklendirme Gücü:** Daha fazla hesaplama gücü ve daha fazla veri, algoritmik bir devrim olmasa bile modellerin yeteneklerini doğrusal bir şekilde artırmaya devam etmektedir.
- [x] **Ampirik Güvenlik:** Geleneksel açıkların yerini, dilin ve görsellerin manipüle edildiği, modelin içsel mantığının hedef alındığı yeni bir siber güvenlik cephesi almıştır.

*Bu teknoloji, muazzam bir potansiyelin yanı sıra dikkatle yönetilmesi gereken riskleri barındıran, insanlık tarihinin en heyecan verici bilişim yolculuğudur.*

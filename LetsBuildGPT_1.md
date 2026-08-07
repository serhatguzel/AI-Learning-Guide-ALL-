# Sıfırdan GPT İnşa Etme ve Yapay Zekanın Gizemli Dünyası

Bugün seninle teknoloji dünyasının en büyük gizemlerinden birinin perdesini aralıyoruz. Karşında devasa, karmaşık bir yapı gibi duran **GPT (Generative Pre-trained Transformer)**, aslında özünde *"Dünyanın En Usta Hikaye Anlatıcısı"* olmaya çalışan bir matematik zinciridir. 

Önüne bir cümle koyduğunda, o cümleyi en mantıklı şekilde nasıl tamamlayacağını bilen, milyonlarca sayfayı hafızasına kazımış bir zekadan bahsediyoruz. Andrej Karpathy'nin efsanevi eğitiminden yola çıkarak, seninle birlikte tıpkı Shakespeare gibi konuşan küçük bir yapay zeka inşa edeceğiz. 

Bu yolculuk bittiğinde, "yapay zeka" senin için korkutucu bir terim değil, parçalarını kendi elinle birleştirdiğin bir lego seti haline gelecek.

---

## 🎯 Bu Raporun 3 Ana Hedefi

1. **Gizemi Çözmek:** Yapay zekanın "büyü" değil, mantıklı bir matematiksel işlem olduğunu anlamak.
2. **Mimarini Tanımak:** Kelimelerin sayıya dönüşüp nasıl anlam kazandığını adım adım görmek.
3. **Büyük Resmi Görmek:** Kendi "Mini-GPT"mizi nasıl eğitebileceğimizi keşfetmek.

Bir aşçının mutfağa girmeden önce malzemeleri tek tek seçmesi gibi, biz de yapay zekamızın ham maddesi olan "veriye" bakarak başlıyoruz. Hazırsan mutfağa girelim!

---

## 📚 1. Malzeme Hazırlığı: Tiny Shakespeare Veri Seti

Yapay zekanın öğrenebilmesi için bir "başucu kitabına" ihtiyacı vardır. Bizim örneğimizde bu, **Tiny Shakespeare** adı verilen, Shakespeare'in tüm eserlerinin birleşimi olan yaklaşık 1 milyon karakterlik bir metin dosyasıdır.

Bilgisayar bu metni okurken, öğrenme sürecini ikiye ayırır. Bu ayrımı anlamak çok kritiktir:

| Veri Bölümü | Oran | Amacı | Benzetme |
| :--- | :--- | :--- | :--- |
| **Eğitim (Train)** | `%90` | Karakter kalıplarını öğrenmek. | Konu anlatımlı ders kitabı. |
| **Doğrulama (Validation)** | `%10` | Modelin performansını ölçmek. | Sürpriz sınav (Ezber mi yaptı, anladı mı?). |

> ⚠️ **Önemli:** Eğer model sadece eğitim verisini "ezberlerse", hiç görmediği bir cümleyi tamamlayamaz. Biz onun ezberlemesini değil, Shakespeare'in mantığını (gramer, kelime seçimi) kavramasını istiyoruz.

---

## 🔢 2. Sayısal Boncuklar: Tokenization ve BPE Devrimi

Bilgisayarlar harfleri veya kelimeleri bizim gibi algılayamazlar; onlar sadece sayılardan anlarlar. Bu yüzden metni parçalara (token) ayırıp sayısal boncuklara çevirmemiz gerekir. Burada çok önemli bir **değiş tokuş (trade-off)** dengesi vardır:

* **Karakter Seviyesi (Küçük Kutu):** Sadece 65 harflik küçük bir boncuk kutumuz olduğunu düşün. "Hi there" yazmak için 8 ayrı boncuk dizmen gerekir `(H-i- -t-h-e-r-e)`. Kutun küçük (hafıza istemez) ama kolyen çok uzun ve ağırdır (bilgisayarı yorar).
* **BPE - Byte Pair Encoding (Büyük Kutu):** Kutuda 50.000 çeşit boncuk vardır. "Hi" veya "there" gibi kalıplar tek bir hazır boncuktur. Kolyen çok kısadır (hızlı işlem) ama o devasa kutuyu saklamak için kocaman bir depo (bellek) gerekir.

### Karakter Seviyesi vs. BPE Karşılaştırması

| Özellik | Karakter Seviyesi (Bizimki) | BPE (Gerçek GPT) |
| :--- | :--- | :--- |
| **Kutu Büyüklüğü (Sözlük)** | 65 Boncuk (Çok küçük) | 50.000 Boncuk (Devasa) |
| **Kolye Uzunluğu (Dizi)** | Çok uzun ve ağır | Kısa ve hafif |
| **Örnek: "Hello"** | H-e-l-l-o (5 parça) | Hello (1 parça) |

---

## 🔮 3. İlk Adım: Bigram Modeli (Acemi Falcı)

En basit yapay zeka modeli **Bigram**'dır. Onu *"sadece bir önceki boncuğa bakarak bir sonrakini tahmin eden acemi bir falcı"* gibi düşünebilirsin.

Örneğin, Shakespeare metninde modelin elinde şu an **'T'** harfi varsa, falcı istatistiklere bakar:
* 'T'den sonra **'H'** gelme ihtimali (özellikle "The" kelimesinden dolayı) **%60**'tır.
* 'T'den sonra **'X'** gelme ihtimali **%0.1**'dir.

> 💡 **Limitasyon:** Bu model sadece bir adım geriye bakabildiği için çok zeki değildir. *"To be or not to..."* dediğinde cümlenin başını hatırlamaz, sadece 'o' harfine bakarak bir şeyler uydurur.

---

## 🧠 4. Beyin Fırtınası: Self-Attention (Kelimelerin Birbirine Soruları)

Sadece bir önceki harfe bakmak yetmez; kelimelerin birbiriyle "konuşması" gerekir. İşte Transformer mimarisinin kalbi burasıdır. Bunu bir sınıftaki öğrencilerin birbirine not defteri uzatması gibi düşünebilirsin.

Bu süreçte 3 anahtar kavram vardır:
* **Query (Sorgu):** "Ben ne arıyorum?" *(Örn: Ben bir fiilim, kendime bir özne arıyorum.)*
* **Key (Anahtar):** "Bende ne bilgi var?" *(Örn: Ben bir ismim, buradayım.)*
* **Value (Değer):** "Eğer benimle anlaşırsan, sana vereceğim asıl mesaj budur."

### Önemli Teknik Detaylar:
* 🎭 **Matematiksel Hile (Maskeleme):** Yapay zeka eğitilirken geleceği görmemesi gerekir. 5. kelimeyi tahmin ediyorsa 6. kelimeye bakması kopya çekmek gibidir. Bunu engellemek için gelecekteki kelimelerin önüne görünmez bir perde *(alt üçgen matris)* çekeriz.
* 🎚️ **Hacim Kontrolü (Scaled Attention):** Eğer öğrenciler birbirine çok bağırırsa (sayılar çok büyürse), model sadece tek bir kişiyi dinler. Sayıları kareköklerine bölerek "ses seviyesini dengeliyoruz" ki model her kelimenin sesini doğru oranda alabilsin.

---

## 📡 5. Çok Kanallı İletişim: Multi-Head Attention ve Feed-Forward

Gerçek bir GPT modelinde sadece bir tane "dikkat" mekanizması yoktur.

| Bölüm | Görevi | Benzetme |
| :--- | :--- | :--- |
| **Multi-Head Attention** | Kelimeler arası iletişimi sağlar. | Bir cümleyi aynı anda inceleyen 4 farklı uzman (biri gramer, biri anlam, biri kafiye uzmanı). |
| **Feed-Forward** | Bilgiyi sindirir. | Uzmanların toplantıdan sonra kendi masalarına çekilip "düşünme süresi" geçirmesi. |

---

## 🏗️ 6. Mimari Destekler: Binanın Yıkılmasını Engellemek

Model çok derinleştiğinde (katlar arttığında) bilgi en alt katlara ulaşırken kaybolabilir. Bunu engellemek için üç mühendislik çözümümüz var:

1. **Residual (Süper Otoban):** Bilginin kaybolmadan en alta akmasını sağlayan bir kestirme yoldur. Karmaşık hesaplamalar içinde ana fikir kaybolmaz.
2. **Layer Normalization (Sosyal Dengeleyici):** Bir partide herkesin aynı tonda konuşmasını sağlamak gibidir. Hiçbir sesin (sayının) çok fazla yükselip diğerlerini bastırmasına izin vermeyen bir terazidir.
3. **Dropout (Zorlu Antrenman):** Modelin bazı nöronlarını rastgele kapatırız. Bu, bir basketbol koçunun oyuncularına "tek gözünüz kapalı antrenman yapın" demesi gibidir. Böylece nöronlar ezberlemeyi keser ve çok daha dayanıklı hale gelirler.

---

## 🦜 7. Büyük Resim: Papağan mı, Asistan mı?

GPT'nin eğitimi iki ana aşamadan oluşur, bu farkı bilmek seni bir uzman yapar:

### 1. Pre-training (Ön Eğitim - Papağan Aşaması)
Model tüm interneti okur. Burada o bir **"Doküman Tamamlayıcıdır"**. Ona bir soru sorarsan cevap vermez, sadece soruyu devam ettirir. *"Nasılsın?"* dersen, o da *"İyi misin, orada mısın?"* diye devam edebilir. Çünkü o sadece kelimelerin peş peşe gelme ihtimalini bilir.

### 2. Fine-tuning (İnce Ayar - Asistan Aşaması)
İşte ChatGPT'nin "insan gibi" olduğu yer burasıdır. İnsanlar ona *"Şöyle sorulara böyle kibar cevaplar ver"* diye örnekler gösterir (RLHF). Burada papağan, ne zaman cevap vermesi gerektiğini öğrenen akıllı bir asistana dönüşür.

---

## 🏁 Kapanış ve Özet

🎉 Artık bir GPT'nin en küçük atomundan (boncuklar) en karmaşık katmanlarına kadar nasıl inşa edildiğini biliyorsun. Bu teknoloji aslında devasa bir olasılık makinesidir; geçmişin izlerini takip ederek geleceği en mantıklı şekilde hayal eder.

### 🔑 Cebindeki 5 Anahtar Bilgi

- [x] **Sayılar Konuşur:** Kelimeler sayısal boncuklara (token) dönüşür, bilgisayar sadece bunları görür.
- [x] **Dikkat (Attention) Her Şeydir:** Kelimeler birbirine "ne arıyorum?" diye sorar ve anlam kazanır.
- [x] **Ses Kontrolü (Scaling):** Matematiksel dengeler sayesinde modelin "aklı karışmaz".
- [x] **Ezberden Kaçış (Dropout):** Bazı yolları kapatarak modelin mantığı kavramasını sağlarız.
- [x] **Papağan vs. Asistan:** Önce dünyayı okur (Pre-train), sonra seninle konuşmayı öğrenir (Fine-tune).

*Yapay zeka dünyası artık senin için bir kara kutu değil, içindeki çarkların nasıl döndüğünü bildiğin harika bir saat!*

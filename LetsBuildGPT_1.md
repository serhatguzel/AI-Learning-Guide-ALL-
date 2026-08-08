# 🧠 Sıfırdan Yapay Zeka Mimarı: Karakter Seviyesinde GPT Masterclass Kılavuzu

Merhaba! Bu rehbere hoş geldin. Bugün seninle birlikte, dünyayı kasıp kavuran ChatGPT gibi devasa sistemlerin mutfağına gireceğiz. Bu yolculuğun sonunda, karmaşık görünen o satırların aslında ne kadar mantıklı olduğunu görecek ve **"Tiny Shakespeare"** veri setini kullanarak kendi "mini-GPT" modelini kodlamış olacaksın. 

Hazırsan başlayalım, boncukları dizmeye başlıyoruz! 🚀

---

## 🧮 1. Dil Modelleri ve Tokenization: Boncuk Kutusu ve Kolyeler

Yapay zeka metinleri bizim gibi okumaz; o, kelimeleri veya harfleri sayılara (tokenlara) dönüştürür. Bunu bir **"Boncuk Kutusu"** gibi düşünebilirsin.

| Özellik | Karakter Seviyesi (Bizimki) | BPE (GPT-4 / TikToken) |
| :--- | :--- | :--- |
| **Kutu Boyutu (Vocab Size)** | Küçük (65 boncuk) | Çok Büyük (50.000+ boncuk) |
| **Kolye Uzunluğu** | Çok uzun (Harf harf) | Kısa ve öz (Parça parça) |
| **Örnek Veri** | `'a'`, `'b'`, `'c'` | `'mer'`, `'haba'`, `'ing'` |

> 💡 **Neden Profesyoneller BPE Seçer?**
> Büyük modeller BPE (Byte Pair Encoding) kullanır çünkü bu yöntem hafızayı daha verimli kullanır. Karakter seviyesinde 1000 harf süren bir metin, BPE ile 200-300 tokena iner. Biz ise **mimarinin mantığını en çıplak haliyle anlamak için** "Karakter Seviyesini" kullanacağız.

---

## 📦 2. Block Size ve Batch Size: PORTAKALı ve GPU Otobanı

Veriyi modele tek tek değil, paketler halinde veririz.

* **Block Size (Bağlam Uzunluğu):** Modelin bir sonraki harfi tahmin etmek için geçmişe ne kadar bakacağıdır. `"PORTAKALı"` örneğini düşün (9 harf). Eğer blok boyutumuz 8 ise, burada tam 8 tane eğitim örneği gizlidir:
  * `P` ➔ `O`
  * `PO` ➔ `R`
  * `PORTAKAL` ➔ `ı`
* **Batch Size (Yığın Boyutu):** GPU'muz çok güçlüdür; aynı anda birden fazla "kolye" dizebilir. Farklı sayfalardan alınmış örnekleri aynı anda işlemeye benzer.

### ⚙️ Teknik Detay: PyTorch Tensör Boyutları
* `B` **(Batch):** Aynı anda işlenen bağımsız örnek sayısı *(Örn: 32)*.
* `T` **(Time / Block Size):** Her örnekteki karakter sayısı *(Örn: 8)*. 

GPU bu `(B, T)` matrisini bir otoban gibi kullanarak tüm tahminleri paralel yapar. Paralel işlem olmazsa eğitim aylar sürerdi!

---

## ⏩ 3. XB ve YB Yapısı: Sağa Kaydırma Hilesi 

Modeli eğitirken ona bir **"Girdi" (`XB`)** bir de **"Hedef" (`YB`)** veririz. `YB`, her zaman `XB`'nin bir adım sağa kaydırılmış halidir.

**Blok Boyutu = 4** olan bir `torch.stack` matris şeması:

| Örnek No | Kelime | `XB` (Girdi Matrisi) | `YB` (Hedef Matrisi) |
| :---: | :--- | :--- | :--- |
| **1** | BİLAL | B, İ, L, A | İ, L, A, L |
| **2** | KEDİM | K, E, D, İ | E, D, İ, M |
| **3** | ELMAS | E, L, M, A | L, M, A, S |

Bu yapı sayesinde modelimiz aynı anda 32 (veya batch size kadar) farklı örneğe bakıp *"Bir sonraki harf ne?"* sorusuna cevap arar.

---

## 📊 4. Bigram Dil Modeli ve Logitler: 65x65'lik Dev Excel Tablosu

En basit modelimiz **Bigram**'dır. Bu model "hafızasızdır". Sadece o anki harfe bakar.

* **Excel Benzetmesi:** Alfabemizdeki 65 karakterin satırlarda ve sütunlarda olduğu dev bir tablo hayal et. Satırda "A" harfi varsa, sütunlarda "B" gelme ihtimali yüksek, "Ğ" gelme ihtimali düşüktür.
* **Logitler `(B, T, C)`:** Modelin her karakter için verdiği "olasılık puanlarıdır". `C` (Channel) boyutu, `vocab_size` yani bizim 65 harfimizi temsil eder.
* ⚠️ **Neden Yetersiz?** Çünkü Bigram geçmişi görmez. Sadece "A"dan sonra ne geleceğine bakar, cümlenin başını hatırlayamaz.

---

## 📉 5. Kayıp (Loss) Analizi: Titiz Öğretmen ve Boyut Düzleştirme

Modelin ne kadar kötü salladığını **Cross Entropy** (Çapraz Entropi) ile ölçeriz. Onu titiz bir öğretmen gibi düşün; yanlış harfe yüksek özgüvenle puan verirsen sana çok düşük not (yüksek kayıp) verir.

* **Teknik Dönüşüm:** PyTorch bu öğretmenin kağıtları düzgün görmesini ister. `(B, T, C)` boyutundaki 3D kutuyu `view(-1, C)` diyerek 2D bir sütuna, hedefleri `(B, T)` ise `view(-1)` ile 1D bir listeye düzleştiririz.

### 📐 Matematiksel Sağlaması
Modelin başında hiçbir şey bilmeyen bir öğrenci tamamen rastgele sallıyorsa, 65 harften birini bilme ihtimali `1/65`'tir. Kayıp formülü (Negatif Log-Likelihood):

$$ -\ln(1/65) \approx 4.17 $$

Eğer ilk denemende kaybın **4.17** civarındaysa, modelin "tertemiz" ve hatasız başladığını anlarsın!

---

## 🎲 6. Üretim Döngüsü (Generate Loop): Kör Yazarın Torbası

Model nasıl metin yazar? Bunu elinde harf boncukları olan bir "Kör Yazar" gibi düşün:

1. **Softmax:** Modelin verdiği kaba puanları (logitleri), toplamı %100 olan "adil yüzdelere" çevirir.
2. **Sampling (`torch.multinomial`):** Torbadan rastgele bir harf çekeriz ama olasılığı yüksek olan boncuğun gelme şansı daha yüksektir.
3. **Concatenation:** Çekilen harfi trenin sonuna yeni bir vagon olarak bağlarız ve süreci baştan başlatırız.

> 🛠 **Mimari Kurnazlık:** Bigram modeli geçmişe ihtiyaç duymaz, sadece son harfe bakar. Ancak biz `generate` fonksiyonuna tüm geçmişi besleriz. Neden mi? Yarın Bigram'ı atıp yerine dev bir Transformer koyduğumuzda API uyumluluğu bozulmasın diye!

---

## 🏋️ 7. Eğitim Döngüsü ve GPU Gücü

Modeli eğitmek için **AdamW** isimli "sakin bir antrenör" kullanırız. Öğrenme oranını (learning rate) `3e-4` (Karpathy Sabiti) gibi küçük "bebek adımları" olarak ayarlarız.

### Sınav Günü Modu (Model Durumları)
| Mod | Komut | Ne Yapar? |
| :--- | :--- | :--- |
| **Eğitim** | `model.train()` | Model hatalarından ders çıkarır, her şeyi not eder. |
| **Sınav** | `model.eval()` | Gradyan hesaplamayı durdurur (`torch.no_grad`), gereksiz yorulmaz. |

> 💻 **GPU vs CPU:** CPU, zor integralleri çözen bir matematik profesörü gibidir. GPU ise aynı anda sadece "2+2" yapabilen ama bunlardan 10.000 tanesini aynı saniyede bitiren ilkokul çocuklarıdır. Dil modellerinde bize o çocuklar lazım!

---

## 👓 8. Matematiksel Dikkat Hilesi: Geleceğe Bakan Kör Gözlükler

GPT'nin asıl büyüsü **Attention** (Dikkat) mekanizmasıdır. Ancak eğitilirken modelin cevap anahtarını (gelecekteki harfleri) görmemesi gerekir.

* **Tril (Alt Üçgen Matrisi):** Bu, modelin önüne takılan bir "Kör Gözlüktür". Üst kısmı eksi sonsuz ile doldurur.
* **Softmax Büyüsü:** Bu eksi sonsuzluklar Softmax'tan geçince tam **"0"** olur. Böylece model geleceği asla göremez.
* **Positional Embeddings (Koltuk Numaraları):** Attention hangisinin 1. hangisinin 10. sırada olduğunu bilmez. Sıra bilgisi kaybolmasın diye her harfe "koltuk numarası" ekleriz.

---

## 🤝 9. Tek Kafalı Öz-Dikkat: Grup Projesi ve Özgeçmişler

Her karakter, diğerlerine kendini tanıtmak ve onlardan bilgi almak ister.

* **Query (Sorgu):** "Ben bunu arıyorum." *(Örn: "Ben bir özneyim, yüklemimi arıyorum.")*
* **Key (Anahtar):** "Bende bu bilgiler var." *(Örn: "Ben bir yüklemim.")*
* **Value (Değer):** "Eğer benimle eşleşirsen, sana bu yeteneğimi veririm."

> ⚖️ **Scaled Attention:** Sorgu ve anahtarı çarptıktan sonra sonucu `sqrt(head_size)` değerine böleriz. Eğer bölmezsek sayılar çok büyür, model tek bir harfe %100 takılı kalır (vanishing gradients) ve öğrenemez.

---

## 🕵️ 10. Çok Kafalı Dikkat ve Feed-Forward

* **Multi-Head Attention (İletişim):** Tek bir kafa yetmez. Biri kafiyeye, diğeri özne-yüklem uyumuna bakan bir "Dedektif Ekibi" hayal et. Her kafa bulgularını birleştirir (`torch.cat`).
* **Feed-Forward / FFN (Hesaplama):** Dedektifler bilgileri topladıktan sonra kendi masalarına çekilir ve sessizce düşünürler. Yapısı basittir: `Linear ➔ ReLU ➔ Linear`.

---

## 🛡️ 11. Derin Ağ Koruma Kalkanı

Model derinleştikçe (6+ katman) sinyaller yorulur. Üç ana korumamız var:

1. **Residual Connections (Kestirme Yollar):** Bilginin doğrudan üst katmanlara akabileceği bir otoyol açar. Gradyanlar en başa kadar yorulmadan gider.
2. **Layer Normalization (Sakinleşme):** Karakterler işlem yapmadan önce derin bir nefes alır. Sayıları belirli bir standartta tutar (Pre-norm).
3. **Dropout:** Bazı nöronları rastgele kapatırız. Bu "zorunlu rotasyon" modelin ezber yapmasını önler.

*Bu üçlü; gradyanların yok olmasını engeller, eğitimi stabil tutar ve ezber yerine mantık kurulmasını sağlar.*

---

## 🦜 12. Pre-training vs. Fine-tuning

* **Pre-training (Ön Eğitim):** Modeli metinle baş başa bırakıp "Bir sonraki harfi tahmin et" diyoruz. Model harika bir *Metin Tamamlayıcı* (Document Completer) olur. Soru sorarsan, soruyu devam ettirir.
* **Fine-tuning (İnce Ayar):** Modelin nazik bir *Asistan* olmasını sağlarız. Soru-cevap örnekleriyle ve RLHF (İnsan geri bildirimi) ile hangi cevabın daha "yardımsever" olduğunu öğretiriz.

---

### 🎉 Tebrikler! 
Artık bir GPT modelinin kalbinde neler döndüğünü, boncukların nasıl dizildiğini ve o devasa matrislerin nasıl konuştuğunu biliyorsun. Şimdi tek yapman gereken: Kodu çalıştır, boncukları diz! Kendi Shakespeare yazan yapay zekan seni bekliyor.
